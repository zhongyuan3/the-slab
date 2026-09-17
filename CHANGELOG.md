# Changelog

All notable changes to this project are documented in this file.

## [0.1.0]

### Added

- Initial release: a `no_std` SLUB-style slab allocator mirroring the
  Linux kernel's `mm/slub.c`, layered on top of `the-buddy-system`.
- `KmemCache` with `uninit`/`init` (static-friendly), `alloc`,
  `alloc_zeroed`, the allocator-free `try_alloc_cached` fast path, `free`,
  `shrink` and `destroy`, plus `validate()` for header, free list, partial
  list and accounting invariants.
- SLUB layout math (`calculate_order`/`calculate_sizes`): object stride,
  header offset, order selection with `min_objects` relaxation, and
  pointer-alignment raising.
- `KmemCache::init_with_min_objects` for an explicit per-slab object
  target (SLUB's `slub_min_objects`).
- `KmallocCaches`, the `kmalloc` size-class facade: one cache per
  `kmalloc_info` class, `kmalloc`/`kzalloc`/`kfree` (pointer inference)/
  `ksize`/`krealloc`, and the large path straight through the page
  allocator with an order tag at the base of the block. Every kmalloc
  slab is one page, so `kfree` finds the cache by page-aligning the
  pointer.
- In-object free lists (`set_freepointer`), the active slab plus doubly
  linked partial list, and `min_partial` empty-slab policy.
- `PageAlloc`/`PhysMap` abstractions with a `DirectMap` linear mapping and
  a `BuddyPages` adapter over the buddy allocator; `Send` so caches can be
  wrapped in a lock.
- Error detection for invalid objects, foreign-cache frees, double frees
  (best effort), corrupted slabs and accounting mismatches.
- Unit, integration (`memblock` → buddy → slab) and documentation tests,
  validated under Miri with Stacked Borrows, Tree Borrows and strict
  provenance.
