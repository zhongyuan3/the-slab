//! The size-class facade, the counterpart of the kernel's `kmalloc`.
//!
//! Mirrors `kmalloc`/`kzalloc`/`kfree`/`ksize`/`krealloc`
//! (`include/linux/slab.h`, `mm/slab_common.c`) on top of the explicit
//! caches:
//!
//! - requests up to the largest size class are served by per-class
//!   [`ObjectCache`]s, one class per entry of [`SIZE_CLASSES`] (the
//!   kernel's `kmalloc_info`);
//! - a class may use a multi-page slab block (for example `kmalloc-4k` and
//!   `kmalloc-8k` on 4 KiB pages);
//! - larger requests go through the page allocator directly
//!   (`alloc_large`), with the allocation order stored in a tag at the
//!   base of the block because there is no `struct page` to hold it.
//!
//! `free` has to find the slab or block behind a pointer without being
//! told the cache. Large blocks carry their tag at the page-aligned base.
//! For slabs, `free` first checks whether the page base is the slab header
//! (the common one-page case), then tries each active cache's block
//! alignment to locate the header of a multi-page slab. This keeps pointer
//! inference working for both single-page and multi-page classes.

use core::alloc::Layout;
use core::mem::size_of;
use core::ptr::NonNull;

use the_mm_core::PageFrame;
use the_mm_core::PhysAddr;

use crate::cache::CacheId;
use crate::cache::ObjectCache;
use crate::error::Error;
use crate::header::SLAB_MAGIC;
use crate::header::SlabHeader;
use crate::pages::PageAlloc;
use crate::pages::PhysMap;

/// The kmalloc size classes in bytes, mirroring the kernel's
/// `kmalloc_info` (including the 96 and 192 intermediates).
///
/// Classes whose layout needs a multi-page block (for example the 4096 and
/// 8192 classes on 4 KiB pages) are still served by slabs; requests beyond
/// the largest class use the large path.
///
/// The alignment a class provides is its natural alignment, the largest
/// power of two dividing the class size (at least pointer alignment):
/// power-of-two classes are aligned to their size and the intermediates to
/// their lowbit (96 to 32, 192 to 64), like SLUB's object placement.
pub const SIZE_CLASSES: [usize; 13] = [
    8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192,
];

/// Natural alignment of a size class.
///
/// SLUB lays objects out at multiples of their size from a page-aligned
/// base, so an object of `size` bytes ends up aligned to the largest power
/// of two dividing `size` (at least pointer alignment). The in-slab header
/// breaks the "multiples from zero" property, so the alignment is applied
/// explicitly when the cache is created.
fn natural_alignment(size: usize) -> usize {
    (size & size.wrapping_neg()).max(core::mem::size_of::<*mut u8>())
}

const CLASS_COUNT: usize = SIZE_CLASSES.len();

const CLASS_NAMES: [&str; CLASS_COUNT] = [
    "kmalloc-8",
    "kmalloc-16",
    "kmalloc-32",
    "kmalloc-64",
    "kmalloc-96",
    "kmalloc-128",
    "kmalloc-192",
    "kmalloc-256",
    "kmalloc-512",
    "kmalloc-1024",
    "kmalloc-2048",
    "kmalloc-4096",
    "kmalloc-8192",
];

/// Magic of the tag at the base of a large allocation.
pub(crate) const LARGE_MAGIC: u32 = 0x1A26_9002;

/// The tag at the base of a large allocation (`alloc_large`).
///
/// The kernel keeps the allocation order in `struct page`; without a
/// vmemmap the block itself carries it. The pointer handed to the caller
/// starts right after the tag, so the tag is never overwritten.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct LargeTag {
    pub(crate) magic: u32,
    pub(crate) order: u8,
    pub(crate) _pad: [u8; 3],
}

/// Size of [`LargeTag`], reserved at the start of every large block.
pub(crate) const LARGE_TAG_SIZE: usize = size_of::<LargeTag>();

/// The allocation kind found by [`KernelHeap::locate`].
enum Located<A> {
    /// The pointer belongs to a slab of the cache at `index`.
    Slab { index: usize },
    /// The pointer belongs to a large page allocation.
    Large { page_phys: A, order: u8 },
}

/// The kernel heap, the counterpart of the kernel's `kmalloc_caches`.
///
/// The set is usable without a global allocator: it is defined in place
/// (usually a `static` behind a lock) and initialized once with
/// [`KernelHeap::init`], following the same `uninit`/`init` pattern as
/// the rest of the crate. All operations take `&mut self` plus the page
/// allocator; locking stays with the caller.
///
/// Size classes may use multi-page slab blocks. `free` and `usable_size`
/// locate the owning allocation from the pointer by checking the large-block
/// tag and the active caches' block alignments; no side table is needed.
#[derive(Debug)]
pub struct KernelHeap {
    caches: [ObjectCache; CLASS_COUNT],
    active: [bool; CLASS_COUNT],
    page_size: usize,
    initialized: bool,
}

impl KernelHeap {
    /// A `const` placeholder for static definitions.
    pub const fn uninit() -> Self {
        Self {
            caches: [const { ObjectCache::uninit() }; CLASS_COUNT],
            active: [false; CLASS_COUNT],
            page_size: 0,
            initialized: false,
        }
    }

    /// Returns `true` once [`KernelHeap::init`] has run.
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the size classes, ascending.
    pub const fn classes() -> &'static [usize] {
        &SIZE_CLASSES
    }

    /// Creates one cache per size class that can fit a slab layout.
    ///
    /// A class may use a multi-page slab block; classes that cannot fit any
    /// supported order are served by the large path.
    ///
    /// Mirrors `create_kmalloc_caches`/`new_kmalloc_cache`.
    ///
    /// # Errors
    ///
    /// [`Error::AlreadyInitialized`] when called twice,
    /// [`Error::InvalidPageSize`] if the allocator reports a bogus page
    /// size, or the first cache initialization error.
    pub fn init<PA: PageAlloc>(&mut self, pages: &PA) -> Result<(), Error> {
        if self.initialized {
            return Err(Error::AlreadyInitialized);
        }

        let page_size = pages.page_size();
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(Error::InvalidPageSize);
        }
        self.page_size = page_size;

        for (index, &size) in SIZE_CLASSES.iter().enumerate() {
            let mut cache = ObjectCache::uninit();
            // `min_objects = 1` keeps slabs small; classes whose layout
            // needs multiple pages (for example kmalloc-4k on 4 KiB pages)
            // are activated too.
            let align = natural_alignment(size).min(page_size);
            let id = CacheId::next();
            match cache.init_with_id_and_min_objects(pages, id, CLASS_NAMES[index], size, align, 1)
            {
                Ok(()) => {
                    self.caches[index] = cache;
                    self.active[index] = true;
                }
                // Classes that do not fit any order are served by the large
                // path.
                Err(Error::ObjectTooLarge) => {}
                Err(error) => return Err(error),
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Returns the index of the smallest class that can hold `size`, or
    /// `None` when the request is larger than every class.
    pub fn class_index(size: usize) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&class| class >= size)
    }

    /// Returns the index of the smallest active class whose size covers
    /// `layout.size()` and whose natural alignment covers
    /// `layout.align()`.
    fn class_index_for_layout(&self, layout: Layout) -> Option<usize> {
        let size = layout.size();
        let align = layout.align();
        (0..CLASS_COUNT).find(|&index| {
            self.active[index]
                && SIZE_CLASSES[index] >= size
                && natural_alignment(SIZE_CLASSES[index]) >= align
        })
    }

    /// Returns the cache of a class, if the class has one.
    pub fn cache(&self, index: usize) -> Option<&ObjectCache> {
        if index < CLASS_COUNT && self.active[index] {
            Some(&self.caches[index])
        } else {
            None
        }
    }

    /// Allocates at least `size` bytes (`size` must be non-zero).
    ///
    /// Mirrors `kmalloc`/`__kmalloc_node`.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidObjectSize`] for a zero size, or the allocator's
    /// [`Error::OutOfMemory`] when neither a slab nor the page allocator
    /// can satisfy the request.
    pub fn alloc<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        size: usize,
    ) -> Result<NonNull<u8>, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        if size == 0 {
            return Err(Error::InvalidObjectSize);
        }

        if let Some(index) = Self::class_index(size) {
            if self.active[index] {
                return self.caches[index].alloc(pages);
            }
        }
        self.alloc_large(pages, size)
    }

    /// Allocates at least `size` bytes with the usable bytes zeroed.
    ///
    /// Mirrors `kzalloc`: slab objects are zeroed up to the class size,
    /// large blocks up to the block size, exactly what [`KernelHeap::usable_size`]
    /// reports.
    pub fn alloc_zeroed<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        size: usize,
    ) -> Result<NonNull<u8>, Error> {
        let object = self.alloc(pages, size)?;
        // SAFETY: `object` was just returned by `alloc`.
        let usable = unsafe { self.usable_size(pages, object)? };
        // SAFETY: the allocation owns `usable` writable bytes.
        unsafe { object.as_ptr().write_bytes(0, usable) };
        Ok(object)
    }

    /// Allocates memory satisfying `layout` (`layout.size()` must be
    /// non-zero).
    ///
    /// The block comes from the smallest class whose object size covers
    /// `layout.size()` and whose natural alignment covers
    /// `layout.align()`, so the returned pointer is aligned to at least
    /// `layout.align()`. Requests that no slab class can satisfy fall back
    /// to the large path when `layout.align() <= 8`; otherwise they are
    /// rejected with [`Error::InvalidAlign`], because the large tag puts
    /// the user pointer on an 8-byte boundary. Use an [`ObjectCache`]
    /// created with the wanted alignment (the counterpart of
    /// `kmem_cache_create(align = ...)`), or a page allocator, for
    /// over-aligned blocks.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidObjectSize`] for a zero size,
    /// [`Error::InvalidAlign`] when neither a slab class nor the large
    /// path can provide the alignment, or the chosen path's allocation
    /// errors.
    pub fn alloc_layout<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        layout: Layout,
    ) -> Result<NonNull<u8>, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        if layout.size() == 0 {
            return Err(Error::InvalidObjectSize);
        }

        if let Some(index) = self.class_index_for_layout(layout) {
            return self.caches[index].alloc(pages);
        }
        if layout.align() <= LARGE_TAG_SIZE {
            return self.alloc_large(pages, layout.size());
        }
        Err(Error::InvalidAlign)
    }

    /// Like [`KernelHeap::alloc_layout`], with the usable bytes zeroed.
    pub fn alloc_zeroed_layout<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        layout: Layout,
    ) -> Result<NonNull<u8>, Error> {
        let object = self.alloc_layout(pages, layout)?;
        // SAFETY: `object` was just returned by `alloc_layout`.
        let usable = unsafe { self.usable_size(pages, object)? };
        // SAFETY: the allocation owns `usable` writable bytes.
        unsafe { object.as_ptr().write_bytes(0, usable) };
        Ok(object)
    }

    /// Resizes an allocation so that it satisfies `new_layout`.
    ///
    /// Like [`KernelHeap::realloc`], but the block is also moved when the
    /// current pointer does not satisfy `new_layout.align()`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation returned by this heap's allocation
    /// methods, and `new_layout` must describe a valid size and alignment.
    ///
    /// # Errors
    ///
    /// Same as [`KernelHeap::alloc_layout`] and
    /// [`KernelHeap::usable_size`].
    pub unsafe fn realloc_layout<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        ptr: NonNull<u8>,
        new_layout: Layout,
    ) -> Result<NonNull<u8>, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        if new_layout.size() == 0 {
            return Err(Error::InvalidObjectSize);
        }

        // SAFETY: the caller guarantees `ptr` is a live allocation.
        let old_size = unsafe { self.usable_size(pages, ptr)? };
        if new_layout.size() <= old_size && ptr.as_ptr().addr() % new_layout.align() == 0 {
            return Ok(ptr);
        }

        let new = self.alloc_layout(pages, new_layout)?;
        // SAFETY: both allocations are valid for their sizes and `new` was
        // just handed out, so it cannot overlap the still-live `ptr`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new.as_ptr(),
                old_size.min(new_layout.size()),
            );
        }
        // SAFETY: the caller guarantees `ptr` is a live allocation.
        unsafe { self.free(pages, ptr)? };
        Ok(new)
    }

    /// Returns an allocation from [`KernelHeap::alloc`] to its slab
    /// or to the page allocator.
    ///
    /// Mirrors `kfree`: large allocations are identified by the tag at the
    /// page-aligned block base; slab objects are found by the page-base
    /// header or by checking each active class's block alignment, so objects
    /// anywhere in a multi-page slab can be freed.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation returned by this heap's allocation
    /// methods and must not have been freed. The mapped allocator memory
    /// around the allocation must be readable and initialized.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidPointer`] for a pointer that does not carry a known
    /// tag (interior or stray pointers), [`Error::CrossCacheFree`] for an
    /// object of a cache that is not part of this set, or the slab's or
    /// page allocator's error.
    pub unsafe fn free<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        ptr: NonNull<u8>,
    ) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }

        let located = self.locate(pages, ptr)?;
        match located {
            Located::Slab { index } => {
                // SAFETY: `locate` matched the slab header's owner to this
                // cache and the caller guarantees `ptr` is live.
                unsafe { self.caches[index].free(pages, ptr) }
            }
            Located::Large { page_phys, order } => pages.free_pages(page_phys, order),
        }
    }

    /// Returns the usable size of an allocation: the size class for slab
    /// objects, the block size minus the tag for large ones.
    ///
    /// Mirrors `ksize`/`__ksize`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation returned by this heap's allocation
    /// methods. The mapped allocator memory around the allocation must be
    /// readable and initialized.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidPointer`] or [`Error::CrossCacheFree`] for pointers
    /// that do not belong to this set.
    pub unsafe fn usable_size<PA: PageAlloc>(
        &self,
        pages: &PA,
        ptr: NonNull<u8>,
    ) -> Result<usize, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }

        let located = self.locate(pages, ptr)?;
        match located {
            Located::Slab { index } => Ok(self.caches[index].layout().object_size()),
            Located::Large { order, .. } => Ok((self.page_size << order) - LARGE_TAG_SIZE),
        }
    }

    /// Resizes an allocation, moving it only when it does not fit.
    ///
    /// Mirrors `krealloc`: shrinking (or growing within the current class
    /// or block) keeps the pointer; otherwise a new allocation is made,
    /// the old bytes are copied and the old allocation is freed. Passing a
    /// null pointer is the caller's `kmalloc`, and `new_size` must be
    /// non-zero.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation returned by this heap's allocation
    /// methods, and `new_size` must be non-zero.
    ///
    /// # Errors
    ///
    /// Same as [`KernelHeap::alloc`] and [`KernelHeap::usable_size`].
    pub unsafe fn realloc<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        ptr: NonNull<u8>,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        if new_size == 0 {
            return Err(Error::InvalidObjectSize);
        }

        // SAFETY: the caller guarantees `ptr` is a live allocation.
        let old_size = unsafe { self.usable_size(pages, ptr)? };
        if new_size <= old_size {
            return Ok(ptr);
        }

        let new = self.alloc(pages, new_size)?;
        // SAFETY: both allocations are valid for their sizes and `new` was
        // just handed out, so it cannot overlap the still-live `ptr`.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new.as_ptr(), old_size.min(new_size));
        }
        // SAFETY: the caller guarantees `ptr` is a live allocation.
        unsafe { self.free(pages, ptr)? };
        Ok(new)
    }

    /// Releases every empty slab of every class, like
    /// [`ObjectCache::shrink`] for each cache.
    pub fn shrink<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        for index in 0..CLASS_COUNT {
            if self.active[index] {
                self.caches[index].shrink(pages)?;
            }
        }
        Ok(())
    }

    /// Releases every slab and consumes the set.
    ///
    /// All objects must have been freed first; see
    /// [`ObjectCache::destroy`].
    pub fn destroy<PA: PageAlloc>(self, pages: &mut PA) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        let active = self.active;
        for (index, cache) in self.caches.into_iter().enumerate() {
            if active[index] {
                cache.destroy(pages)?;
            }
        }
        Ok(())
    }

    /// Verifies the invariants of every class cache.
    pub fn validate<M: PhysMap>(&self, map: &M) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        for index in 0..CLASS_COUNT {
            if self.active[index] {
                self.caches[index].validate(map)?;
            }
        }
        Ok(())
    }

    /// Allocates from the page allocator for a request larger than every
    /// usable slab class (`alloc_large`).
    fn alloc_large<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        size: usize,
    ) -> Result<NonNull<u8>, Error> {
        let needed = size
            .checked_add(LARGE_TAG_SIZE)
            .ok_or(Error::OutOfMemory)?
            .div_ceil(self.page_size);
        let max_pages = 1usize << pages.max_order();
        if needed > max_pages {
            return Err(Error::OutOfMemory);
        }
        let order = needed.next_power_of_two().trailing_zeros() as u8;

        let block = pages.alloc_pages(order)?;
        let base = pages.phys_to_virt(block);
        // SAFETY: the block is page aligned and at least
        // `LARGE_TAG_SIZE` bytes, so the tag write is aligned and in
        // bounds.
        unsafe {
            base.cast::<LargeTag>().write(LargeTag {
                magic: LARGE_MAGIC,
                order,
                _pad: [0; 3],
            });
        }
        // SAFETY: the tag occupies the first `LARGE_TAG_SIZE` bytes.
        Ok(unsafe { base.add(LARGE_TAG_SIZE) })
    }

    /// Locates the allocation behind `ptr`.
    ///
    /// Large allocations are recognized by the tag at the page-aligned block
    /// base. Slab objects are recognized by finding the active cache whose
    /// block alignment produces a valid slab header owned by that cache;
    /// this supports multi-page slabs where the object is not in the first
    /// page.
    ///
    /// The caller must uphold the live-allocation contract of
    /// [`KernelHeap::free`].
    fn locate<PA: PageAlloc>(
        &self,
        pages: &PA,
        ptr: NonNull<u8>,
    ) -> Result<Located<PA::Addr>, Error> {
        let phys = pages.try_virt_to_phys(ptr).ok_or(Error::InvalidPointer)?;
        let page_size = PA::Addr::try_from_usize(self.page_size).ok_or(Error::InternalError)?;
        let page_phys = PA::Addr::align_down(phys, page_size);
        let page = pages
            .try_phys_to_virt(page_phys)
            .ok_or(Error::InvalidPointer)?;

        // SAFETY: `page` is page aligned and within the mapping. The
        // caller's live-allocation contract makes the surrounding allocator
        // memory readable.
        let magic = unsafe { page.cast::<u32>().read() };

        // Fast path: the object is in the first page of a slab, so the page
        // base carries the slab header.
        if magic == SLAB_MAGIC {
            // SAFETY: the magic matched; the page starts with a slab header.
            let header = unsafe { &*page.cast::<SlabHeader>().as_ptr() };
            if let Some(index) = self.cache_index(header.owner) {
                let cache = &self.caches[index];
                if header.order == cache.layout().order() {
                    if let Some(slab_bytes) = PA::Addr::try_from_usize(cache.layout().slab_bytes())
                    {
                        if PA::Addr::align_down(phys, slab_bytes) == page_phys {
                            return Ok(Located::Slab { index });
                        }
                    }
                }
            }
        }

        // Multi-page slabs: try each active cache's block alignment. The
        // object may be in any page of the block, so the page base is not
        // necessarily the slab header.
        for (index, cache) in self.caches.iter().enumerate() {
            if !self.active[index] {
                continue;
            }
            let Some(slab_bytes) = PA::Addr::try_from_usize(cache.layout().slab_bytes()) else {
                continue;
            };
            let base = PA::Addr::align_down(phys, slab_bytes);
            let Some(slab) = pages.try_phys_to_virt(base) else {
                continue;
            };
            // SAFETY: the caller guarantees `ptr` is a live allocation; the
            // aligned block base is allocator memory.
            let header = unsafe { &*slab.cast::<SlabHeader>().as_ptr() };
            if header.magic == SLAB_MAGIC
                && header.owner == cache.id()
                && header.order == cache.layout().order()
            {
                return Ok(Located::Slab { index });
            }
        }

        // Large allocations carry their tag at the page-aligned block base.
        // Check this after the slab scan so that object data in a later page
        // of a multi-page slab cannot be mistaken for a large tag.
        if magic == LARGE_MAGIC {
            // SAFETY: the magic matched; read the tag from the block base.
            let (order, expected) = unsafe {
                let tag = &*page.cast::<LargeTag>().as_ptr();
                (tag.order, page.add(LARGE_TAG_SIZE))
            };
            if ptr == expected && order <= pages.max_order() {
                return Ok(Located::Large { page_phys, order });
            }
        }

        if magic == SLAB_MAGIC {
            Err(Error::CrossCacheFree)
        } else {
            Err(Error::InvalidPointer)
        }
    }

    /// Returns the index of the active cache whose id matches `owner`.
    fn cache_index(&self, owner: CacheId) -> Option<usize> {
        self.caches
            .iter()
            .enumerate()
            .find_map(|(index, cache)| (self.active[index] && cache.id() == owner).then_some(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_index_picks_the_smallest_class() {
        assert_eq!(KernelHeap::class_index(1), Some(0));
        assert_eq!(KernelHeap::class_index(8), Some(0));
        assert_eq!(KernelHeap::class_index(9), Some(1));
        assert_eq!(KernelHeap::class_index(96), Some(4));
        assert_eq!(KernelHeap::class_index(97), Some(5));
        assert_eq!(KernelHeap::class_index(8192), Some(12));
        assert_eq!(KernelHeap::class_index(8193), None);
    }

    #[test]
    fn natural_alignment_is_the_lowbit_of_the_class() {
        let expected = [
            (8usize, 8usize),
            (16, 16),
            (32, 32),
            (64, 64),
            (96, 32),
            (128, 128),
            (192, 64),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
        ];
        for (size, align) in expected {
            assert_eq!(natural_alignment(size), align, "class {size}");
        }
    }

    #[test]
    fn uninit_is_const_and_inert() {
        const PLACEHOLDER: KernelHeap = KernelHeap::uninit();
        assert!(!PLACEHOLDER.is_initialized());
        assert_eq!(PLACEHOLDER.page_size, 0);
        assert_eq!(KernelHeap::classes().len(), CLASS_COUNT);
    }

    #[test]
    fn tag_layouts_start_with_their_magic() {
                // `free` dispatches on the first word of a page, so both the slab
        // header and the large tag must start with their magic.
        assert_eq!(core::mem::offset_of!(SlabHeader, magic), 0);
        assert_eq!(core::mem::offset_of!(LargeTag, magic), 0);
        assert_ne!(LARGE_MAGIC, SLAB_MAGIC);
        assert_eq!(size_of::<LargeTag>(), LARGE_TAG_SIZE);
    }
}
