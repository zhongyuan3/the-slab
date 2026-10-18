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

pub use crate::cache::CacheId;
pub use crate::cache::ObjectCache;
pub use crate::error::Error;
pub use crate::heap::KernelHeap;
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
pub mod buddy;
pub mod cache;
pub mod error;
pub mod heap;
pub mod layout;
pub mod pages;

pub(crate) mod header;
