# the-slab

A `no_std` reimplementation of the Linux kernel's [SLUB] slab allocator
(`mm/slub.c`), layered on top of [the-buddy-system].

Each `ObjectCache` carves page blocks requested from the page allocator into
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
  `ObjectCache::init`, following the same `uninit`/`init` pattern as the
  buddy allocator.
- Kernel-shaped layers: `ObjectCache` over a `PageAlloc` abstract
  (`BuddyPages` borrows the buddy allocator, `Zone` owns it so the page
  allocator itself can live in a `static`), over a `PhysMap` (`DirectMap`
  covers the common direct-map case and is also `static`-friendly).
- SLUB algorithms and states: `calculate_order`/`calculate_sizes` layout,
  in-object free lists, active slab + partial list, `min_partial`, and
  `alloc_zeroed`.
- A kernel heap: `KernelHeap` creates one cache per size class
  (`kmalloc_info`), serves larger requests straight from the page
  allocator (`alloc_large`), and provides `alloc`/`alloc_zeroed`/`free`
  (pointer inference)/`usable_size`/`realloc`.
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
use the_slab::{BuddyPages, DirectMap, ObjectCache};

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

let mut cache = ObjectCache::uninit();
cache.init(&pages, "widgets", 24, 8).unwrap();

let object = cache.alloc(&mut pages).unwrap();
assert_eq!(object.as_ptr() as usize % 8, 0);
cache.free(&mut pages, object).unwrap();
cache.destroy(&mut pages).unwrap();
```

## Kernel heap

`KernelHeap` mirrors the kernel's size-class API (`kmalloc`) on top of
the same page allocator. It is defined in place and initialized once, and
`free` finds the owning cache or large block from the pointer alone:

```rust
use the_slab::KernelHeap;

let mut heap = KernelHeap::uninit();
heap.init(&pages).unwrap();

let object = heap.alloc(&mut pages, 100).unwrap();
assert!(heap.usable_size(&pages, object).unwrap() >= 100);
heap.free(&mut pages, object).unwrap();

// Larger requests bypass the slabs and use the page allocator directly.
let block = heap.alloc_zeroed(&mut pages, 5000).unwrap();
heap.free(&mut pages, block).unwrap();

heap.shrink(&mut pages).unwrap();
heap.destroy(&mut pages).unwrap();
```

Every kmalloc slab is exactly one page, so `free` can page-align the
pointer to reach the slab header; requests that would need a multi-page
slab (the kernel's two-page `kmalloc-4k`/`kmalloc-8k` on 4 KiB pages) go
through the large path instead, where a tag at the page-aligned base of
the block records the allocation order.

Allocations carry the kernel's natural alignment: a power-of-two class
returns that alignment and the intermediates at least their lowbit (96 →
32, 192 → 64). For an explicit alignment use `alloc_layout(Layout)`,
which picks the smallest class covering both size and alignment:

```rust
use core::alloc::Layout;

let layout = Layout::from_size_align(100, 32).unwrap();
let object = heap.alloc_layout(&mut pages, layout).unwrap();
assert_eq!(object.as_ptr() as usize % 32, 0);
```

Requests that no class can serve use the large path when the alignment is
at most 8 and are rejected with `Error::InvalidAlign` otherwise; an
`ObjectCache` created with the wanted alignment (the counterpart of
`kmem_cache_create(align = ...)`), or the page allocator, serves
over-aligned blocks.

## Page allocator statics

Caches and the kernel heap are typically statics, while `BuddyPages`
borrows the buddy allocator and therefore cannot be one. `Zone` owns a
buddy arena plus its `DirectMap`, so the page allocator can be a static
too:

```rust
use std::sync::Mutex;
use the_slab::Zone;

static ZONE: Mutex<Zone<usize, 11>> = Mutex::new(Zone::uninit());
static CACHE: Mutex<the_slab::ObjectCache> = Mutex::new(the_slab::ObjectCache::uninit());

// After memory discovery, once the vmemmap is mapped:
let mut zone = ZONE.lock().unwrap();
// SAFETY: the descriptors and the mapping are valid for the rest of the
// system, and all access goes through the lock.
unsafe { zone.init(vmemmap, nr_pages, base, page_size, virt_base) }.unwrap();
zone.buddy_mut().free_memblock(&memblock).unwrap();
drop(zone);

// Caches borrow the zone for the duration of each call.
CACHE.lock().unwrap().init(&*ZONE.lock().unwrap(), "widgets", 24, 8).unwrap();
let object = CACHE.lock().unwrap().alloc(&mut *ZONE.lock().unwrap()).unwrap();
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
- multi-page kmalloc classes; the kernel's two-page `kmalloc-4k` and
  `kmalloc-8k` are served by the large path — `TODO(kmalloc)`

## License

MIT. See [LICENSE](LICENSE).
