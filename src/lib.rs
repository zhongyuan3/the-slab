//! A `no_std` reimplementation of the Linux kernel's SLUB-style slab
//! allocator (`mm/slub.c`).
//!
//! Each [`ObjectCache`] manages fixed-size objects carved out of page blocks
//! requested from a [`PageAlloc`]. A slab block starts with a slab header
//! and is followed by objects on an aligned stride; free objects store the
//! next free list pointer inside themselves, exactly like SLUB's
//! `set_freepointer`. The cache keeps one active slab (`cpu_slab`) plus a
//! partial list (`kmem_cache_node`) and hands empty slabs back to the page
//! allocator once `min_partial` is exceeded.
//!
//! # Layering
//!
//! ```text
//! ObjectCache::alloc/free            kmem_cache_alloc / kmem_cache_free
//!     |
//! PageAlloc                         alloc_pages / __free_pages
//!     |
//! PhysMap                            the kernel's direct map
//!     |
//! the-buddy-system (optional)        the buddy page allocator
//! ```
//!
//! The slab core depends only on the [`PageAlloc`]/[`PhysMap`] traits and on
//! the shared [`the_mm_core`] address traits. It does not depend on a
//! concrete page allocator:
//!
//! - with the `buddy` feature, `BuddyPages` and `Zone` adapt
//!   `the_buddy_system::Buddy`;
//! - without it, implement [`PageAlloc`] over your own page source. Any
//!   allocator, including a kernel-specific one, can serve slab caches.
//!
//! The boot handoff is explicit: feed `Zone::free_ranges` with a boot
//! allocator's free ranges (for example a memblock's). The slab crate has no
//! memblock dependency, and the buddy crate has none either.
//!
//! Locking stays with the caller: the cache core is `&mut self`, and
//! [`ObjectCache::try_alloc_cached`] is the allocator-free fast path. Each
//! cache stores a stable [`CacheId`] in its slab headers, so moving an
//! initialized cache or [`KernelHeap`] does not invalidate its slabs.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "buddy")] {
//! use core::alloc::Layout;
//! use core::ptr::NonNull;
//!
//! use the_buddy_system::Buddy;
//! use the_buddy_system::Page;
//! use the_slab::BuddyPages;
//! use the_slab::DirectMap;
//! use the_slab::ObjectCache;
//!
//! const PAGE: usize = 0x1000;
//! const PAGES: usize = 64;
//! // `NR_PAGE_ORDERS` free areas, so the largest order (`Buddy::MAX_ORDER`)
//! // is 3 and the biggest block is 8 pages.
//! const NR_PAGE_ORDERS: usize = 4;
//!
//! // A real allocation standing in for physical memory.
//! let layout = Layout::from_size_align(PAGES * PAGE, PAGE << (NR_PAGE_ORDERS - 1)).unwrap();
//! // SAFETY: `layout` has a non-zero size.
//! let memory = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
//! let base = memory.as_ptr() as usize;
//!
//! let mut descriptors = vec![Page::EMPTY; PAGES];
//! let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
//! buddy.free_range(base, base + PAGES * PAGE).unwrap();
//!
//! // SAFETY: the identity mapping covers the allocation.
//! let map = unsafe { DirectMap::new(base, memory) };
//! let mut pages = BuddyPages::new(&mut buddy, map);
//!
//! let mut cache = ObjectCache::uninit();
//! cache.init(&pages, "widgets", 24, 8).unwrap();
//!
//! let object = cache.alloc(&mut pages).unwrap();
//! assert_eq!(object.as_ptr() as usize % 8, 0);
//! // SAFETY: `object` came from this cache and is still live.
//! unsafe { cache.free(&mut pages, object) }.unwrap();
//! cache.destroy(&mut pages).unwrap();
//! # }
//! ```
//!
//! # Kernel references
//!
//! - `mm/slub.c`: `calculate_order`, `calculate_sizes`,
//!   `set_freepointer`/`get_freepointer`, `allocate_slab`, `__slab_alloc`,
//!   `get_partial`, `slab_free`, `kmem_cache_shrink`, `kmem_cache_destroy`
//! - `include/linux/slab.h`: `kmem_cache_alloc`, `kmem_cache_free`,
//!   `kmalloc`, `kzalloc`, `kfree`, `ksize`, `krealloc`
//! - `include/linux/slub_def.h`: `struct kmem_cache`, `min_partial`
//! - `include/linux/mmzone.h`: `struct zone`
//! - `mm/slab_common.c`: `kmalloc_info`, `create_kmalloc_caches`
//!
//! # Roadmap
//!
//! The extension points are marked with `TODO(name)` comments in the
//! source:
//!
//! - `TODO(percpu)`: per-CPU magazines and the lock-free `cpu_slab` fast
//!   path.
//! - `TODO(color)`: slab coloring to spread objects across cache lines.
//! - `TODO(poison)`: `SLAB_POISON`/red zones and full object checking.
//! - `TODO(ctor)`: constructors and destructors.

#![no_std]

/// Errors returned by the slab allocator.
///
/// The variants mirror the failure modes SLUB reports through
/// `slab_err`/`object_err` (`mm/slub.c`) and the allocation failure of
/// `slab_alloc_node`.
///
/// [`ObjectCache::init`]: crate::ObjectCache::init
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The page allocator has no free block of the requested order.
    OutOfMemory,
    /// An operation that needs a layout was attempted before
    /// [`ObjectCache::init`].
    Uninitialized,
    /// [`ObjectCache::init`] was called on a cache that is already
    /// initialized.
    AlreadyInitialized,
    /// The object size is zero or smaller than a free list pointer.
    InvalidObjectSize,
    /// The alignment is zero, not a power of two, larger than the page
    /// size, or not satisfiable by any available slab class.
    InvalidAlign,
    /// The page size is zero or not a power of two.
    InvalidPageSize,
    /// No buddy order up to `max_order` can hold even one object.
    ObjectTooLarge,
    /// The pointer does not denote an object of this cache: it is
    /// misaligned, outside the object area, or points into memory that is
    /// not a slab header.
    InvalidPointer,
    /// The pointer belongs to a slab of a different cache.
    CrossCacheFree,
    /// The object is already free.
    DoubleFree,
    /// Slab metadata is corrupted (bad magic/owner/order, broken free list
    /// or partial list).
    CorruptSlab,
    /// Bookkeeping does not match the slabs that can be walked.
    CountMismatch,
    /// The underlying page allocator failed for a reason other than
    /// running out of memory.
    PageAlloc,
    /// An invariant was violated; indicates a bug in the implementation.
    InternalError,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Error::OutOfMemory => "Out of memory",
            Error::Uninitialized => "Object cache is not initialized",
            Error::AlreadyInitialized => "Object cache is already initialized",
            Error::InvalidObjectSize => "Invalid object size",
            Error::InvalidAlign => "Invalid alignment",
            Error::InvalidPageSize => "Invalid page size",
            Error::ObjectTooLarge => "Object does not fit a slab block",
            Error::InvalidPointer => "Invalid object pointer",
            Error::CrossCacheFree => "Object belongs to another cache",
            Error::DoubleFree => "Object is already free",
            Error::CorruptSlab => "Corrupted slab metadata",
            Error::CountMismatch => "Slab accounting mismatch",
            Error::PageAlloc => "Page allocator failure",
            Error::InternalError => "Internal error",
        };
        write!(f, "{s}")
    }
}

impl core::error::Error for Error {}

pub use crate::cache::CacheId;
pub use crate::cache::ObjectCache;
pub use crate::heap::KernelHeap;
pub use crate::heap::SIZE_CLASSES;
pub use crate::layout::SlabLayout;
pub use crate::pages::DirectMap;
pub use crate::pages::PageAlloc;
pub use crate::pages::PhysMap;
pub use the_mm_core;
pub use the_mm_core::PageFrame;
pub use the_mm_core::PhysAddr;

#[cfg(feature = "buddy")]
pub use crate::buddy::BuddyPages;
#[cfg(feature = "buddy")]
pub use crate::buddy::Zone;

#[cfg(feature = "buddy")]
mod buddy;
mod cache;
mod heap;
mod layout;
mod pages;

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::ToString;

    use super::*;

    #[test]
    fn error_is_copy_and_eq() {
        let error = Error::OutOfMemory;
        let copy = error;
        assert_eq!(error, copy);
        assert_ne!(Error::OutOfMemory, Error::DoubleFree);
    }

    #[test]
    fn display_covers_all_variants() {
        assert_eq!(Error::OutOfMemory.to_string().as_str(), "Out of memory");
        assert_eq!(
            Error::Uninitialized.to_string().as_str(),
            "Object cache is not initialized"
        );
        assert_eq!(
            Error::AlreadyInitialized.to_string().as_str(),
            "Object cache is already initialized"
        );
        assert_eq!(
            Error::InvalidObjectSize.to_string().as_str(),
            "Invalid object size"
        );
        assert_eq!(
            Error::InvalidAlign.to_string().as_str(),
            "Invalid alignment"
        );
        assert_eq!(
            Error::InvalidPageSize.to_string().as_str(),
            "Invalid page size"
        );
        assert_eq!(
            Error::ObjectTooLarge.to_string().as_str(),
            "Object does not fit a slab block"
        );
        assert_eq!(
            Error::InvalidPointer.to_string().as_str(),
            "Invalid object pointer"
        );
        assert_eq!(
            Error::CrossCacheFree.to_string().as_str(),
            "Object belongs to another cache"
        );
        assert_eq!(
            Error::DoubleFree.to_string().as_str(),
            "Object is already free"
        );
        assert_eq!(
            Error::CorruptSlab.to_string().as_str(),
            "Corrupted slab metadata"
        );
        assert_eq!(
            Error::CountMismatch.to_string().as_str(),
            "Slab accounting mismatch"
        );
        assert_eq!(
            Error::PageAlloc.to_string().as_str(),
            "Page allocator failure"
        );
        assert_eq!(Error::InternalError.to_string().as_str(), "Internal error");
    }
}
