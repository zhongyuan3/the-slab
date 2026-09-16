//! A `no_std` reimplementation of the Linux kernel's SLUB-style slab
//! allocator (`mm/slub.c`), layered on top of the buddy page allocator.
//!
//! Each [`KmemCache`] manages fixed-size objects carved out of page blocks
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
//! KmemCache::alloc/free            kmem_cache_alloc / kmem_cache_free
//!     |
//! PageAlloc (BuddyPages)           alloc_pages / __free_pages
//!     |
//! PhysMap (DirectMap)              the kernel's direct map
//!     |
//! the-buddy-system                 the buddy page allocator
//! ```
//!
//! [`BuddyPages`] pairs a [`Buddy`](the_buddy_system::Buddy) allocator with
//! a [`DirectMap`], so a kernel only has to supply its direct-map offset.
//! Locking stays with the caller, as in the buddy crate: the cache core is
//! `&mut self`, and [`KmemCache::try_alloc_cached`] is the
//! allocator-free fast path.
//!
//! # Examples
//!
//! ```
//! use core::alloc::Layout;
//! use core::ptr::NonNull;
//!
//! use the_buddy_system::Buddy;
//! use the_buddy_system::Page;
//! use the_slab::BuddyPages;
//! use the_slab::DirectMap;
//! use the_slab::KmemCache;
//!
//! const PAGE: usize = 0x1000;
//! const PAGES: usize = 64;
//! const MAX_ORDER: usize = 4;
//!
//! // A real allocation standing in for physical memory.
//! let layout = Layout::from_size_align(PAGES * PAGE, PAGE << (MAX_ORDER - 1)).unwrap();
//! // SAFETY: `layout` has a non-zero size.
//! let memory = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
//! let base = memory.as_ptr() as usize;
//!
//! let mut descriptors = vec![Page::EMPTY; PAGES];
//! let mut buddy = Buddy::<usize, MAX_ORDER>::new(base, PAGE, &mut descriptors).unwrap();
//! buddy.free_range(base, base + PAGES * PAGE).unwrap();
//!
//! // SAFETY: the identity mapping covers the allocation.
//! let map = unsafe { DirectMap::new(base, memory) };
//! let mut pages = BuddyPages::new(&mut buddy, map);
//!
//! let mut cache = KmemCache::uninit();
//! cache.init(&pages, "widgets", 24, 8).unwrap();
//!
//! let object = cache.alloc(&mut pages).unwrap();
//! assert_eq!(object.as_ptr() as usize % 8, 0);
//! cache.free(&mut pages, object).unwrap();
//! cache.destroy(&mut pages).unwrap();
//! ```
//!
//! # Kernel references
//!
//! - `mm/slub.c`: `calculate_order`, `calculate_sizes`,
//!   `set_freepointer`/`get_freepointer`, `allocate_slab`, `__slab_alloc`,
//!   `get_partial`, `slab_free`, `kmem_cache_shrink`, `kmem_cache_destroy`
//! - `include/linux/slab.h`: `kmem_cache_alloc`, `kmem_cache_free`
//! - `include/linux/slub_def.h`: `struct kmem_cache`, `min_partial`
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
//! - `TODO(kmalloc)`: size classes, `kmalloc`/`kfree`, and the large
//!   allocation path through the page allocator.

#![no_std]

pub use crate::cache::KmemCache;
pub use crate::error::Error;
pub use crate::layout::SlabLayout;
pub use crate::pages::BuddyPages;
pub use crate::pages::DirectMap;
pub use crate::pages::PageAlloc;
pub use crate::pages::PhysMap;

pub mod cache;
pub mod error;
pub mod layout;
pub mod pages;

pub(crate) mod header;
