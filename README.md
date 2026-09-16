# the-slab

A `no_std` reimplementation of the Linux kernel's [SLUB] slab allocator
(`mm/slub.c`), layered on top of [the-buddy-system].

Each `KmemCache` carves page blocks requested from the page allocator into
fixed-size objects. A slab block starts with a header and is followed by
objects on an aligned stride; free objects store the free list pointer
inside themselves (`set_freepointer`), exactly like SLUB. The cache keeps
one active slab (`cpu_slab`) plus a doubly linked partial list
(`kmem_cache_node`) and returns empty slabs to the page allocator once
`min_partial` is exceeded.

[SLUB]: https://www.kernel.org/doc/html/latest/core-api/memory-allocation.html
[the-buddy-system]: https://crates.io/crates/the-buddy-system

## Features

- `no_std`, no global allocator: the cache is defined in place (a stack
  local, a `Box`, or a `static`) and initialized once with
  `KmemCache::init`, following the same `uninit`/`init` pattern as the
  buddy allocator.
- Kernel-shaped layers: `KmemCache` over a `PageAlloc` abstract
  (`BuddyPages` adapts the buddy allocator), over a `PhysMap`
  (`DirectMap` covers the common direct-map case).
- SLUB algorithms and states: `calculate_order`/`calculate_sizes` layout,
  in-object free lists, active slab + partial list, `min_partial`, and
  `alloc_zeroed`.
- `try_alloc_cached`, the allocator-free fast path a kernel runs without
  the zone lock; all other methods take `&mut self` and leave locking to
  the caller.
- Robustness checks: header magic and owning-cache checks reject foreign
  or stray pointers, best-effort double free detection, alignment
  enforcement, and a `validate()` invariant checker (headers, free lists,
  partial list, accounting) mirroring SLUB's `check_object`.
- `Send` for lock wrapping; caches can live in `static`s.
- Validated under Miri with both aliasing models and strict provenance.

## Usage

```rust
use core::alloc::Layout;
use core::ptr::NonNull;

use the_buddy_system::{Buddy, Page};
use the_slab::{BuddyPages, DirectMap, KmemCache};

const PAGE: usize = 0x1000;
const PAGES: usize = 64;
const MAX_ORDER: usize = 4;

// A real allocation standing in for physical memory.
let layout = Layout::from_size_align(PAGES * PAGE, PAGE << (MAX_ORDER - 1)).unwrap();
let memory = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
let base = memory.as_ptr() as usize;

let mut descriptors = vec![Page::EMPTY; PAGES];
let mut buddy = Buddy::<usize, MAX_ORDER>::new(base, PAGE, &mut descriptors).unwrap();
buddy.free_range(base, base + PAGES * PAGE).unwrap();

// SAFETY: the identity mapping covers the allocation.
let map = unsafe { DirectMap::new(base, memory) };
let mut pages = BuddyPages::new(&mut buddy, map);

let mut cache = KmemCache::uninit();
cache.init(&pages, "widgets", 24, 8).unwrap();

let object = cache.alloc(&mut pages).unwrap();
assert_eq!(object.as_ptr() as usize % 8, 0);
cache.free(&mut pages, object).unwrap();
cache.destroy(&mut pages).unwrap();
```

## Boot handoff

The intended kernel sequence is `memblock` → buddy → slab: reserve the
kernel image in `memblock`, hand the remaining memory to the buddy
allocator with `Buddy::free_memblock`, then serve object allocations from
caches on top. See `tests/integration.rs` for the full flow, including
that memory reserved during boot is never handed out.

## Testing

`cargo test` runs the unit, integration and documentation tests. The slab
free list lives in object memory and all metadata access goes through raw
pointers, so the suite is also validated under Miri with both aliasing
models:

```sh
cargo +nightly miri test --lib --tests
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly miri test --lib --tests
```

## Roadmap

Not implemented yet; the extension points are marked with `TODO(name)`
comments in the source:

- per-CPU magazines and the lock-free `cpu_slab` fast path —
  `TODO(percpu)`
- slab coloring — `TODO(color)`
- `SLAB_POISON`/red zones and full object checking — `TODO(poison)`
- constructors and destructors — `TODO(ctor)`
- `kmalloc` size classes, `kfree` and the large allocation path —
  `TODO(kmalloc)`

## License

MIT. See [LICENSE](LICENSE).
