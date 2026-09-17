//! The size-class facade, the counterpart of the kernel's `kmalloc`.
//!
//! Mirrors `kmalloc`/`kzalloc`/`kfree`/`ksize`/`krealloc`
//! (`include/linux/slab.h`, `mm/slab_common.c`) on top of the explicit
//! caches:
//!
//! - requests up to the largest usable *single-page* class are served by
//!   per-class [`ObjectCache`]s, one class per entry of [`SIZE_CLASSES`]
//!   (the kernel's `kmalloc_info`);
//! - larger requests go through the page allocator directly
//!   (`alloc_large`), with the allocation order stored in a tag at the
//!   base of the block because there is no `struct page` to hold it.
//!
//! `free` has to find the slab or block behind a pointer without being
//! told the cache. That is why every kmalloc slab is exactly one page and
//! every large block carries its tag at the page-aligned base: `free`
//! only has to look at the first word of the page containing the pointer.
//! Slab allocation of multi-page blocks therefore stays with the explicit
//! [`ObjectCache`] API (the kernel's two-page classes, `kmalloc-4k` and
//! `kmalloc-8k` on 4 KiB pages, are served by the large path here).

use core::mem::size_of;
use core::ptr::NonNull;

use the_buddy_system::PageFrame;
use the_memblock::PhysAddr;

use crate::cache::ObjectCache;
use crate::error::Error;
use crate::header::SLAB_MAGIC;
use crate::header::SlabHeader;
use crate::pages::PageAlloc;
use crate::pages::PhysMap;

/// The kmalloc size classes in bytes, mirroring the kernel's
/// `kmalloc_info` (including the 96 and 192 intermediates).
///
/// Sizes above the largest class, and classes whose slab would need more
/// than one page, are served by the large path.
pub const SIZE_CLASSES: [usize; 13] = [
    8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192,
];

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

/// The kernel heap, the counterpart of the kernel's `kmalloc_caches`.
///
/// The set is usable without a global allocator: it is defined in place
/// (usually a `static` behind a lock) and initialized once with
/// [`KernelHeap::init`], following the same `uninit`/`init` pattern as
/// the rest of the crate. All operations take `&mut self` plus the page
/// allocator; locking stays with the caller.
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

    /// Creates one cache per size class, where the layout fits a single
    /// page; other classes fall back to the large path.
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
            // `min_objects = 1` keeps every kmalloc slab on a single page,
            // which `free` relies on to find the cache by page-aligning
            // the object pointer.
            match cache.init_with_min_objects(pages, CLASS_NAMES[index], size, 8, 1) {
                Ok(()) if cache.layout().order() == 0 => {
                    self.caches[index] = cache;
                    self.active[index] = true;
                }
                // Classes that would need a multi-page slab, or that do
                // not fit at all, are served by the large path.
                Ok(()) | Err(Error::ObjectTooLarge) => {}
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
        let usable = self.usable_size(pages, object)?;
        // SAFETY: the allocation owns `usable` writable bytes.
        unsafe { object.as_ptr().write_bytes(0, usable) };
        Ok(object)
    }

    /// Returns an allocation from [`KernelHeap::alloc`] to its slab
    /// or to the page allocator.
    ///
    /// Mirrors `kfree`: the pointer's page carries either a slab header or
    /// a large tag, which identifies the owning cache or the block order.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidPointer`] for a pointer that does not carry a known
    /// tag (interior or stray pointers), [`Error::CrossCacheFree`] for an
    /// object of a cache that is not part of this set, or the slab's or
    /// page allocator's error.
    pub fn free<PA: PageAlloc>(&mut self, pages: &mut PA, ptr: NonNull<u8>) -> Result<(), Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }

        let (page, page_phys, magic) = self.page_tag(pages, ptr);
        match magic {
            SLAB_MAGIC => {
                // SAFETY: the magic matched, so the page starts with a
                // slab header.
                let owner = unsafe { (*page.cast::<SlabHeader>().as_ptr()).owner };
                let index = self.cache_index(owner).ok_or(Error::CrossCacheFree)?;
                self.caches[index].free(pages, ptr)
            }
            LARGE_MAGIC => {
                // SAFETY: the magic matched, so the page starts with a
                // large tag written by `alloc_large`.
                let (order, expected) = unsafe {
                    let tag = &*page.cast::<LargeTag>().as_ptr();
                    (tag.order, page.add(LARGE_TAG_SIZE))
                };
                if ptr != expected || order > pages.max_order() {
                    return Err(Error::InvalidPointer);
                }
                pages.free_pages(page_phys, order)
            }
            _ => Err(Error::InvalidPointer),
        }
    }

    /// Returns the usable size of an allocation: the size class for slab
    /// objects, the block size minus the tag for large ones.
    ///
    /// Mirrors `ksize`/`__ksize`.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KernelHeap::init`],
    /// [`Error::InvalidPointer`] or [`Error::CrossCacheFree`] for pointers
    /// that do not belong to this set.
    pub fn usable_size<PA: PageAlloc>(&self, pages: &PA, ptr: NonNull<u8>) -> Result<usize, Error> {
        if !self.initialized {
            return Err(Error::Uninitialized);
        }

        let (page, _, magic) = self.page_tag(pages, ptr);
        match magic {
            SLAB_MAGIC => {
                // SAFETY: the magic matched, so the page starts with a
                // slab header.
                let owner = unsafe { (*page.cast::<SlabHeader>().as_ptr()).owner };
                let index = self.cache_index(owner).ok_or(Error::CrossCacheFree)?;
                Ok(self.caches[index].layout().object_size())
            }
            LARGE_MAGIC => {
                // SAFETY: the magic matched, so the page starts with a
                // large tag written by `alloc_large`.
                let (order, expected) = unsafe {
                    let tag = &*page.cast::<LargeTag>().as_ptr();
                    (tag.order, page.add(LARGE_TAG_SIZE))
                };
                if ptr != expected || order > pages.max_order() {
                    return Err(Error::InvalidPointer);
                }
                Ok((self.page_size << order) - LARGE_TAG_SIZE)
            }
            _ => Err(Error::InvalidPointer),
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
    /// # Errors
    ///
    /// Same as [`KernelHeap::alloc`] and [`KernelHeap::usable_size`].
    pub fn realloc<PA: PageAlloc>(
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

        let old_size = self.usable_size(pages, ptr)?;
        if new_size <= old_size {
            return Ok(ptr);
        }

        let new = self.alloc(pages, new_size)?;
        // SAFETY: both allocations are valid for their sizes and `new` was
        // just handed out, so it cannot overlap the still-live `ptr`.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new.as_ptr(), old_size.min(new_size));
        }
        self.free(pages, ptr)?;
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

    /// Locates the page containing `ptr` and reads its first word, which
    /// is the slab magic or the large tag magic.
    fn page_tag<PA: PageAlloc>(&self, pages: &PA, ptr: NonNull<u8>) -> (NonNull<u8>, PA::Addr, u32) {
        let phys = pages.virt_to_phys(ptr);
        let page_size = PA::Addr::from_usize(self.page_size);
        let page_phys = PA::Addr::align_down(phys, page_size);
        let page = pages.phys_to_virt(page_phys);
        // SAFETY: `page` is page aligned. Its first word is written by
        // this module for live allocations and is arbitrary (initialized)
        // memory otherwise; the magic checks reject the latter.
        let magic = unsafe { page.cast::<u32>().read() };
        (page, page_phys, magic)
    }

    /// Returns the index of the active cache whose address matches
    /// `owner`.
    fn cache_index(&self, owner: *const ()) -> Option<usize> {
        self.caches.iter().enumerate().find_map(|(index, cache)| {
            let address = core::ptr::from_ref(cache).cast::<()>();
            (self.active[index] && address == owner).then_some(index)
        })
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
