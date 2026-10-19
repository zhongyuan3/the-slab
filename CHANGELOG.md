# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Breaking changes

- **Pointer-taking methods are now `unsafe`.** `ObjectCache::free`,
  `KernelHeap::free`, `KernelHeap::usable_size`, `KernelHeap::realloc` and
  `KernelHeap::realloc_layout` require the caller to pass a live allocation
  from the matching allocator. The methods read slab/large-allocation
  metadata through the pointer, so this precondition can no longer be left
  implicit in a safe function. `alloc`, `alloc_zeroed`, `alloc_layout` and
  the allocation-free fast path remain safe.

### Added

- **Optional `buddy` feature.** `BuddyPages`, `Zone` and the buddy error
  mapping now live in the `buddy` module behind `features = ["buddy"]`. The slab core (`ObjectCache`, `KernelHeap`,
  `SlabLayout`, `PageAlloc`, `PhysMap`, `DirectMap`) has no buddy or
  memblock dependency.
- `Zone::free_range` and `Zone::free_ranges` pass boot memory through to the
  owned arena, so the handoff is one explicit call on the page allocator.
- `the-mm-core` is re-exported together with its `PhysAddr` and
  `PageFrame` traits.

### Changed

- **Flattened the module layout.** Public items are exported only from the
  crate root. The `error` and `header` modules were removed: `Error` lives
  in `lib.rs`, and `SlabHeader`/`SLAB_MAGIC` live in `cache.rs`.
  `SIZE_CLASSES` is now re-exported at the crate root.
- **The slab core is allocator-neutral.** The `pages` module now contains
  only the `PageAlloc`/`PhysMap` traits and `DirectMap`; the buddy adapters
  moved to the `buddy` module. `cache` and `heap` import the address traits
  from `the-mm-core`.
- `the-buddy-system` is an optional dependency; the default feature set is
  empty. The slab library has no memblock dependency at all.
- **PFN conversions moved to `PageFrame`.** The shared `PageFrame` trait
  now owns `phys_to_pfn`, `pfn_to_phys` and `pfn_ceil` in addition to the
  `usize` index conversions. Custom `PageFrame` implementations must
  provide the new methods; the `the-slab` API itself is unchanged.
- `PhysMap` gained checked `try_phys_to_virt` and `try_virt_to_phys`
  methods. `DirectMap` overrides them to reject addresses below the mapping
  instead of panicking, and `ObjectCache::free`/`KernelHeap::free` use them
  so a stray pointer below the map is reported as `Error::InvalidPointer`.
- **Move-safe cache identity.** `ObjectCache` now stores a stable `CacheId`
  in each slab header instead of comparing a freshly computed `self`
  address on every free. `CacheId` is allocated from a process-wide counter
  by `init`, or supplied explicitly with `init_with_id`; `KernelHeap`
  assigns one per class. Moving an initialized cache or heap therefore keeps
  frees working. `CacheId` is re-exported at the crate root; the default id
  source is a process-wide `AtomicUsize` counter, while `init_with_id`
  accepts an explicit id.
- The boot handoff is explicit:
  `zone.free_ranges(memblock.free_mem_ranges(flags))`. The previous
  `Zone::free_memblock` convenience method was removed along with the
  `memblock` feature, so memblock and buddy stay fully decoupled.
- **Multi-page kmalloc classes.** `KernelHeap::init` now activates size
  classes whose layout needs a multi-page slab block (for example
  `kmalloc-4096` and `kmalloc-8192` on 4 KiB pages). Full slabs are linked
  into a per-cache full list, so `free` and `usable_size` can locate an
  object by walking each active cache's slab lists (active, partial and
  full) and checking the block range; no arbitrary memory scan is needed.
  `validate` and `destroy` now cover full slabs as well. The large path
  remains for requests beyond the largest class and for alignments that no
  class can satisfy.
- `KernelHeap` now stores its caches as `[Option<ObjectCache>; CLASS_COUNT]`
  instead of a cache array plus a parallel `active` bitmap, so the two
  states cannot diverge.

### Compatibility

- Enable `features = ["buddy"]` to keep using `BuddyPages` and `Zone`.
- Replace `zone.free_memblock(&mb)` with
  `zone.free_ranges(mb.free_mem_ranges(flags))`.

## [0.1.0]

### Added

- Initial release: a `no_std` SLUB-style slab allocator mirroring the
  Linux kernel's `mm/slub.c`, layered on top of `the-buddy-system`.
- `ObjectCache` with `uninit`/`init` (static-friendly), `alloc`,
  `alloc_zeroed`, the allocator-free `try_alloc_cached` fast path, `free`,
  `shrink` and `destroy`, plus `validate()` for header, free list, partial
  list and accounting invariants.
- SLUB layout math (`calculate_order`/`calculate_sizes`): object stride,
  header offset, order selection with `min_objects` relaxation, and
  pointer-alignment raising.
- `ObjectCache::init_with_min_objects` for an explicit per-slab object
  target (SLUB's `slub_min_objects`).
- `KernelHeap`, the `kmalloc` size-class facade: one cache per
  `kmalloc_info` class, `alloc`/`alloc_zeroed`/`free` (pointer inference)/
  `usable_size`/`realloc`, and the large path straight through the page
  allocator with an order tag at the base of the block. Every kmalloc
  slab is one page, so `free` finds the cache by page-aligning the
  pointer. Each class carries its natural alignment (power-of-two classes
  to their size, 96 to 32, 192 to 64), matching SLUB's object placement.
- `KernelHeap::alloc_layout`/`alloc_zeroed_layout`/`realloc_layout` take
  a `Layout` and pick the smallest class covering both the size and the
  alignment; over-aligned requests that no class can serve report
  `Error::InvalidAlign` (use an `ObjectCache` with the wanted alignment,
  or the page allocator).
- In-object free lists (`set_freepointer`), the active slab plus doubly
  linked partial list, and `min_partial` empty-slab policy.
- `PageAlloc`/`PhysMap` abstractions with a `DirectMap` linear mapping and
  a `BuddyPages` adapter over the buddy allocator; `Send` so caches can be
  wrapped in a lock.
- `Zone`, a self-contained page allocator that owns a `Buddy<'static>`
  arena plus its `DirectMap` (`uninit`/`init`, `from_raw_parts`,
  `buddy`/`buddy_mut`), so the page allocator itself can live in a
  `static` while caches borrow it per call. `DirectMap` gained
  `uninit`/`init`/`is_initialized` for the same reason.
- Error detection for invalid objects, foreign-cache frees, double frees
  (best effort), corrupted slabs and accounting mismatches.
- Unit, integration (`memblock` → buddy → slab) and documentation tests,
  validated under Miri with Stacked Borrows, Tree Borrows and strict
  provenance.
