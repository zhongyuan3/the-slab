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
//! For slabs, `free` walks each active cache's slab lists (active, partial
//! and full) and checks which slab block contains the pointer. This keeps
//! pointer inference working for both single-page and multi-page classes
//! without scanning arbitrary memory.

use core::alloc::Layout;
use core::mem::size_of;
use core::ptr::NonNull;

use the_mm_core::PageFrame;
use the_mm_core::PhysAddr;

use crate::cache::CacheId;
use crate::cache::ObjectCache;
use crate::error::Error;
use crate::header::SLAB_MAGIC;
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
/// locate the owning allocation from the pointer by walking the active
/// caches' slab lists (including full slabs) and checking the block range,
/// then the large-block tag; no side table is needed.
#[derive(Debug)]
pub struct KernelHeap {
    caches: [Option<ObjectCache>; CLASS_COUNT],
    page_size: usize,
    initialized: bool,
}

impl KernelHeap {
    /// A `const` placeholder for static definitions.
    pub const fn uninit() -> Self {
        Self {
            caches: [const { None }; CLASS_COUNT],
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
                    self.caches[index] = Some(cache);
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
            self.caches[index].is_some()
                && SIZE_CLASSES[index] >= size
                && natural_alignment(SIZE_CLASSES[index]) >= align
        })
    }

    /// Returns the cache of a class, if the class has one.
    pub fn cache(&self, index: usize) -> Option<&ObjectCache> {
        self.caches.get(index).and_then(|cache| cache.as_ref())
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
            if let Some(cache) = self.caches.get_mut(index).and_then(|cache| cache.as_mut()) {
                return cache.alloc(pages);
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
            if let Some(cache) = self.caches.get_mut(index).and_then(|cache| cache.as_mut()) {
                return cache.alloc(pages);
            }
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
    /// Mirrors `kfree`: slab objects are found by walking each cache's slab
    /// lists (active, partial and full) and checking the block range; large
    /// allocations are identified by the tag at the page-aligned block base.
    /// Objects anywhere in a multi-page slab can be freed.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation returned by this heap's allocation
    /// methods and must not have been freed.
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
                let Some(cache) = self.caches[index].as_mut() else {
                    return Err(Error::InternalError);
                };
                // SAFETY: `locate` matched this cache and the caller
                // guarantees `ptr` is live.
                unsafe { cache.free(pages, ptr) }
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
    /// methods.
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
            Located::Slab { index } => {
                let cache = self.caches[index].as_ref().ok_or(Error::InternalError)?;
                Ok(cache.layout().object_size())
            }
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
        for cache in self.caches.iter_mut().flatten() {
            cache.shrink(pages)?;
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
        for cache in self.caches.into_iter().flatten() {
            cache.destroy(pages)?;
        }
        Ok(())
    }

    /// Verifies the invariants of every class cache.
    pub fn validate<M: PhysMap>(&self, map: &M) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }
        for cache in self.caches.iter().flatten() {
            cache.validate(map)?;
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
    /// Slab objects are found by walking each active cache's slab lists
    /// (active, partial and full) and checking which slab block contains the
    /// physical address. This supports multi-page slabs without reading
    /// untrusted memory or relying on the page base. Large allocations are
    /// recognized by the tag at the page-aligned block base.
    ///
    /// The caller must uphold the live-allocation contract of
    /// [`KernelHeap::free`].
    fn locate<PA: PageAlloc>(
        &self,
        pages: &PA,
        ptr: NonNull<u8>,
    ) -> Result<Located<PA::Addr>, Error> {
        let phys = pages.try_virt_to_phys(ptr).ok_or(Error::InvalidPointer)?;

        for (index, cache) in self.caches.iter().enumerate() {
            let Some(cache) = cache.as_ref() else {
                continue;
            };
            let Some(slab_bytes) = PA::Addr::try_from_usize(cache.layout().slab_bytes()) else {
                continue;
            };
            for slab in cache.slabs() {
                let Some(base) = pages.try_virt_to_phys(slab.cast()) else {
                    continue;
                };
                if phys >= base && phys - base < slab_bytes {
                    return Ok(Located::Slab { index });
                }
            }
        }

        // No slab matched: a large allocation carries its tag at the
        // page-aligned block base.
        let page_size = PA::Addr::try_from_usize(self.page_size).ok_or(Error::InternalError)?;
        let page_phys = PA::Addr::align_down(phys, page_size);
        let page = pages
            .try_phys_to_virt(page_phys)
            .ok_or(Error::InvalidPointer)?;
        // SAFETY: `page` is page aligned and within the mapping. A live
        // large allocation has its tag there.
        let magic = unsafe { page.cast::<u32>().read() };

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::SlabHeader;

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
