//! Integration tests: `memblock` → buddy → slab, over real memory.

use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::Mutex;

use the_buddy_system::Buddy;
use the_buddy_system::Page;
use the_memblock::flags::MemblockFlags;
use the_memblock::memblock::Memblock;
use the_slab::BuddyPages;
use the_slab::DirectMap;
use the_slab::Error;
use the_slab::KernelHeap;
use the_slab::ObjectCache;
use the_slab::Zone;

const PAGE: usize = 0x1000;
const PAGES: usize = 64;
// `NR_PAGE_ORDERS` free areas: valid orders are 0..=3 (`Buddy::MAX_ORDER`
// is 3) and the largest block is 8 pages.
const NR_PAGE_ORDERS: usize = 4;
// Largest block, i.e. 1 << Buddy::MAX_ORDER pages.
const BLOCK_BYTES: usize = PAGE << (NR_PAGE_ORDERS - 1);

/// A block of real memory, block-aligned and zeroed, standing in for
/// physical memory.
struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Memory {
    fn new(pages: usize) -> Self {
        let layout = Layout::from_size_align(pages * PAGE, BLOCK_BYTES).expect("valid layout");
        // SAFETY: `layout` has a non-zero size.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) }).expect("allocation failed");
        // SAFETY: the allocation spans `pages * PAGE` writable bytes.
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0, pages * PAGE) };
        Self { ptr, layout }
    }

    fn base(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: the allocation came from `alloc` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[test]
fn slab_over_buddy_end_to_end() {
    let memory = Memory::new(PAGES);
    let base = memory.base();

    let mut descriptors = vec![Page::EMPTY; PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + PAGES * PAGE).unwrap();
    assert_eq!(buddy.nr_free(), PAGES);

    // SAFETY: the identity mapping covers the backing allocation, which
    // outlives the page allocator and every cache using it.
    let map = unsafe { DirectMap::new(base, memory.ptr) };
    let mut pages = BuddyPages::new(&mut buddy, map);

    let mut cache = ObjectCache::uninit();
    cache.init(&pages, "widgets", 24, 8).unwrap();

    // Allocate a large run and check the objects are distinct, aligned and
    // writable.
    let mut live = Vec::new();
    for index in 0..1000usize {
        let object = cache.alloc(&mut pages).unwrap();
        assert_eq!(object.as_ptr() as usize % 8, 0);
        // SAFETY: the object holds 24 writable bytes.
        unsafe { object.as_ptr().write_bytes(index as u8, 24) };
        live.push(object);
    }
    let mut addresses: Vec<usize> = live.iter().map(|object| object.as_ptr() as usize).collect();
    addresses.sort_unstable();
    addresses.dedup();
    assert_eq!(addresses.len(), live.len());
    cache.validate(&pages).unwrap();

    // Free half, then the rest.
    for object in live.drain(..500) {
        cache.free(&mut pages, object).unwrap();
    }
    cache.validate(&pages).unwrap();
    for object in live {
        cache.free(&mut pages, object).unwrap();
    }
    assert_eq!(cache.nr_free_objects(), cache.nr_objects());
    cache.validate(&pages).unwrap();

    cache.shrink(&mut pages).unwrap();
    assert_eq!(cache.nr_slabs(), 0);
    cache.destroy(&mut pages).unwrap();
}

#[test]
fn boot_handoff_from_memblock_to_slab() {
    let memory = Memory::new(PAGES);
    let base = memory.base();

    // The boot story: memblock knows the memory, the kernel image is
    // reserved, and the rest is handed to the buddy allocator.
    let mut mb = Memblock::<usize, 16>::new();
    mb.add(base, PAGES * PAGE, MemblockFlags::NONE).unwrap();
    mb.reserve_kern(base, 2 * PAGE).unwrap();

    let mut descriptors = vec![Page::EMPTY; PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_memblock(&mb).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut cache = ObjectCache::uninit();
        cache.init(&pages, "handoff", 128, 8).unwrap();

        // Allocate until the arena is exhausted; the reserved kernel range
        // must never be handed out.
        let mut live = Vec::new();
        while let Ok(object) = cache.alloc(&mut pages) {
            let address = object.as_ptr() as usize;
            assert!(!(base..base + 2 * PAGE).contains(&address));
            live.push(object);
        }
        assert!(live.len() > 100);
        cache.validate(&pages).unwrap();

        for object in live {
            cache.free(&mut pages, object).unwrap();
        }
        cache.shrink(&mut pages).unwrap();
        assert_eq!(cache.nr_slabs(), 0);
        cache.destroy(&mut pages).unwrap();
    }

    // Every page that was not reserved is back in the buddy allocator.
    assert_eq!(buddy.nr_free(), PAGES - 2);
}

/// A cache defined in a static, as a kernel would.
static CACHE: Mutex<ObjectCache> = Mutex::new(ObjectCache::uninit());

#[test]
fn cache_can_be_defined_in_a_static() {
    let guard = CACHE.lock().unwrap();
    assert!(!guard.is_initialized());
    assert_eq!(guard.nr_slabs(), 0);
}

#[test]
fn heap_across_classes_roundtrip() {
    const MEMORY_PAGES: usize = 128;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        // Requests up to the largest single-page class come from slabs.
        let mut live = Vec::new();
        for &size in &[1usize, 8, 9, 64, 96, 100, 192, 256, 1024, 2048] {
            for round in 0..4u8 {
                let object = heap.alloc(&mut pages, size).unwrap();
                let usable = heap.usable_size(&pages, object).unwrap();
                assert!(usable >= size);
                // SAFETY: the object owns `usable` writable bytes.
                unsafe { object.as_ptr().write_bytes(round, usable) };
                live.push(object);
            }
        }

        let mut addresses: Vec<usize> = live.iter().map(|o| o.as_ptr() as usize).collect();
        addresses.sort_unstable();
        addresses.dedup();
        assert_eq!(addresses.len(), live.len());
        heap.validate(&pages).unwrap();

        for object in live {
            heap.free(&mut pages, object).unwrap();
        }
        heap.validate(&pages).unwrap();

        heap.shrink(&mut pages).unwrap();
        for index in 0..KernelHeap::classes().len() {
            if let Some(cache) = heap.cache(index) {
                assert_eq!(cache.nr_slabs(), 0);
            }
        }
        heap.destroy(&mut pages).unwrap();
    }

    // Every page went back to the buddy allocator.
    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn heap_large_uses_the_page_allocator() {
    const MEMORY_PAGES: usize = 64;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        for size in [4096usize, 5000, 16384] {
            let object = heap.alloc(&mut pages, size).unwrap();
            let usable = heap.usable_size(&pages, object).unwrap();
            assert!(usable >= size);
            assert!(object.as_ptr() as usize > base);
            assert!((object.as_ptr() as usize) < base + MEMORY_PAGES * PAGE);
            // SAFETY: the object owns `usable` writable bytes.
            unsafe { object.as_ptr().write_bytes(0xab, usable) };
            heap.free(&mut pages, object).unwrap();
        }

        heap.validate(&pages).unwrap();
        heap.destroy(&mut pages).unwrap();
    }

    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn heap_alloc_zeroed_and_realloc() {
    const MEMORY_PAGES: usize = 64;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        // Zeroing covers the whole usable allocation.
        let zeroed = heap.alloc_zeroed(&mut pages, 200).unwrap();
        let usable = heap.usable_size(&pages, zeroed).unwrap();
        // SAFETY: the object is allocated and readable.
        let bytes = unsafe { core::slice::from_raw_parts(zeroed.as_ptr(), usable) };
        assert!(bytes.iter().all(|byte| *byte == 0));
        heap.free(&mut pages, zeroed).unwrap();

        // Growing copies the old bytes; shrinking keeps the pointer.
        let object = heap.alloc(&mut pages, 100).unwrap();
        // SAFETY: the object owns at least 100 writable bytes.
        unsafe { object.as_ptr().write_bytes(0x5a, 100) };
        let grown = heap.realloc(&mut pages, object, 3000).unwrap();
        assert!(heap.usable_size(&pages, grown).unwrap() >= 3000);
        // SAFETY: `realloc` copied the old contents.
        let copied = unsafe { core::slice::from_raw_parts(grown.as_ptr(), 100) };
        assert!(copied.iter().all(|byte| *byte == 0x5a));
        let same = heap.realloc(&mut pages, grown, 64).unwrap();
        assert_eq!(same, grown);
        heap.free(&mut pages, same).unwrap();

        heap.validate(&pages).unwrap();
        heap.destroy(&mut pages).unwrap();
    }

    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn heap_allocations_are_naturally_aligned() {
    const MEMORY_PAGES: usize = 64;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        // Power-of-two classes return their own alignment; the 96 and 192
        // intermediates at least their lowbit.
        for (size, align) in [
            (8usize, 8usize),
            (16, 16),
            (64, 64),
            (96, 32),
            (192, 64),
            (256, 256),
            (2048, 2048),
        ] {
            let object = heap.alloc(&mut pages, size).unwrap();
            assert_eq!(object.as_ptr().addr() % align, 0, "size {size}");
            heap.free(&mut pages, object).unwrap();
        }

        // A request that rounds up to a larger class gets its alignment.
        let object = heap.alloc(&mut pages, 100).unwrap(); // class 128
        assert_eq!(object.as_ptr().addr() % 128, 0);
        heap.free(&mut pages, object).unwrap();

        heap.shrink(&mut pages).unwrap();
        heap.destroy(&mut pages).unwrap();
    }

    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn alloc_layout_honors_size_and_alignment() {
    const MEMORY_PAGES: usize = 128;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        // (size, align, expected class): the smallest class covering both.
        for (size, align, class) in [
            (1usize, 8usize, 8usize),
            (64, 64, 64),
            (100, 32, 128),
            (4, 64, 64),
            (200, 128, 256),
        ] {
            let layout = Layout::from_size_align(size, align).unwrap();
            let object = heap.alloc_layout(&mut pages, layout).unwrap();
            assert_eq!(object.as_ptr().addr() % align, 0, "{size}/{align}");
            assert_eq!(
                heap.usable_size(&pages, object).unwrap(),
                class,
                "{size}/{align}"
            );
            heap.free(&mut pages, object).unwrap();
        }

        // Requests larger than every class use the large path when the
        // alignment fits the tag's 8-byte boundary.
        let layout = Layout::from_size_align(10_000, 8).unwrap();
        let object = heap.alloc_layout(&mut pages, layout).unwrap();
        assert!(heap.usable_size(&pages, object).unwrap() >= 10_000);
        heap.free(&mut pages, object).unwrap();

        // Over-aligned requests that no class can serve are rejected:
        // class 4096 would need a multi-page slab on 4 KiB pages.
        let layout = Layout::from_size_align(64, 4096).unwrap();
        assert!(matches!(
            heap.alloc_layout(&mut pages, layout),
            Err(Error::InvalidAlign)
        ));

        // Zero-sized layouts are rejected like `alloc(0)`.
        let layout = Layout::from_size_align(0, 8).unwrap();
        assert!(matches!(
            heap.alloc_layout(&mut pages, layout),
            Err(Error::InvalidObjectSize)
        ));

        heap.validate(&pages).unwrap();
        heap.destroy(&mut pages).unwrap();
    }

    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn alloc_zeroed_layout_and_realloc_layout() {
    const MEMORY_PAGES: usize = 64;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut heap = KernelHeap::uninit();
        heap.init(&pages).unwrap();

        // Zeroing covers the usable allocation.
        let layout = Layout::from_size_align(100, 32).unwrap();
        let object = heap.alloc_zeroed_layout(&mut pages, layout).unwrap();
        let usable = heap.usable_size(&pages, object).unwrap();
        // SAFETY: the object is allocated and readable.
        let bytes = unsafe { core::slice::from_raw_parts(object.as_ptr(), usable) };
        assert!(bytes.iter().all(|byte| *byte == 0));
        heap.free(&mut pages, object).unwrap();

        // A stronger alignment is honored, keeping the old contents.
        let small = heap
            .alloc_layout(&mut pages, Layout::from_size_align(4, 8).unwrap())
            .unwrap();
        // SAFETY: the object owns at least 4 writable bytes.
        unsafe { small.as_ptr().write_bytes(0x5a, 4) };
        let moved = heap
            .realloc_layout(&mut pages, small, Layout::from_size_align(4, 64).unwrap())
            .unwrap();
        assert_eq!(moved.as_ptr().addr() % 64, 0);
        // SAFETY: `realloc_layout` copied the old contents.
        let copied = unsafe { core::slice::from_raw_parts(moved.as_ptr(), 4) };
        assert!(copied.iter().all(|byte| *byte == 0x5a));

        // A satisfied alignment keeps the pointer.
        let same = heap
            .realloc_layout(&mut pages, moved, Layout::from_size_align(2, 64).unwrap())
            .unwrap();
        assert_eq!(same, moved);
        heap.free(&mut pages, same).unwrap();

        heap.validate(&pages).unwrap();
        heap.destroy(&mut pages).unwrap();
    }

    assert_eq!(buddy.nr_free(), MEMORY_PAGES);
}

#[test]
fn free_rejects_bad_pointers() {
    const MEMORY_PAGES: usize = 32;
    let memory = Memory::new(MEMORY_PAGES);
    let base = memory.base();
    let mut descriptors = vec![Page::EMPTY; MEMORY_PAGES];
    let mut buddy = Buddy::<usize, NR_PAGE_ORDERS>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + MEMORY_PAGES * PAGE).unwrap();

    // SAFETY: the identity mapping covers the backing allocation.
    let map = unsafe { DirectMap::new(base, memory.ptr) };
    let mut pages = BuddyPages::new(&mut buddy, map);

    let mut heap = KernelHeap::uninit();
    heap.init(&pages).unwrap();

    let object = heap.alloc(&mut pages, 64).unwrap();

    // An interior pointer is not on the object grid.
    // SAFETY: the arithmetic stays inside the object.
    let interior = unsafe { NonNull::new(object.as_ptr().add(8)).unwrap() };
    assert!(matches!(
        heap.free(&mut pages, interior),
        Err(Error::InvalidPointer)
    ));

    // A pointer into memory that never carried a tag.
    // SAFETY: the arithmetic stays inside the backing allocation.
    let stray = unsafe { NonNull::new(memory.ptr.as_ptr().add(20 * PAGE)).unwrap() };
    assert!(matches!(
        heap.free(&mut pages, stray),
        Err(Error::InvalidPointer)
    ));

    // An object of a cache outside the kernel heap.
    let mut foreign = ObjectCache::uninit();
    foreign.init(&pages, "foreign", 64, 8).unwrap();
    let foreign_object = foreign.alloc(&mut pages).unwrap();
    assert!(matches!(
        heap.free(&mut pages, foreign_object),
        Err(Error::CrossCacheFree)
    ));
    foreign.free(&mut pages, foreign_object).unwrap();
    foreign.destroy(&mut pages).unwrap();

    heap.free(&mut pages, object).unwrap();
    heap.shrink(&mut pages).unwrap();
    heap.destroy(&mut pages).unwrap();
}

/// The kernel heap defined in a static, as a kernel would.
static HEAP: Mutex<KernelHeap> = Mutex::new(KernelHeap::uninit());

#[test]
fn heap_can_be_defined_in_a_static() {
    let guard = HEAP.lock().unwrap();
    assert!(!guard.is_initialized());
    assert!(guard.cache(0).is_none());
}

#[test]
fn static_zone_object_cache_and_heap_share_the_page_allocator() {
    // Everything a kernel keeps for the whole system lives in statics: the
    // zone (page allocator), an object cache and the kernel heap.
    static ZONE: Mutex<Zone<usize, NR_PAGE_ORDERS>> = Mutex::new(Zone::uninit());
    static CACHE: Mutex<ObjectCache> = Mutex::new(ObjectCache::uninit());
    static STATIC_HEAP: Mutex<KernelHeap> = Mutex::new(KernelHeap::uninit());

    // The descriptors and the backing memory must outlive the static zone.
    // The descriptors are leaked (they are reachable through the zone),
    // and the memory owner is forgotten at the end of the test so the
    // backing allocation stays alive as well.
    let descriptors: &'static mut [Page] =
        Box::leak(vec![Page::EMPTY; PAGES].into_boxed_slice());
    let memory = Memory::new(PAGES);
    let base = memory.base();
    let ptr = NonNull::new(descriptors.as_mut_ptr()).unwrap();

    {
        let mut zone = ZONE.lock().unwrap();
        assert!(!zone.is_initialized());
        // SAFETY: the leaked descriptors and backing memory live for the
        // rest of the process, and all access goes through the lock.
        unsafe { zone.init(ptr, PAGES, base, PAGE, memory.ptr) }.unwrap();
        zone.buddy_mut()
            .free_range(base, base + PAGES * PAGE)
            .unwrap();
        assert_eq!(zone.buddy().nr_free(), PAGES);
    }

    // The caches borrow the zone for the duration of each call only;
    // `&*guard` hands out the `Zone` behind the mutex. Lock order follows
    // the kernel: cache first, then zone.
    {
        let mut cache = CACHE.lock().unwrap();
        let mut heap = STATIC_HEAP.lock().unwrap();
        let zone = ZONE.lock().unwrap();
        cache.init(&*zone, "static", 24, 8).unwrap();
        heap.init(&*zone).unwrap();
    }

    let object = CACHE
        .lock()
        .unwrap()
        .alloc(&mut *ZONE.lock().unwrap())
        .unwrap();
    assert_eq!(object.as_ptr() as usize % 8, 0);

    let block = STATIC_HEAP
        .lock()
        .unwrap()
        .alloc(&mut *ZONE.lock().unwrap(), 200)
        .unwrap();
    assert!(
        STATIC_HEAP
            .lock()
            .unwrap()
            .usable_size(&*ZONE.lock().unwrap(), block)
            .unwrap()
            >= 200
    );

    CACHE
        .lock()
        .unwrap()
        .free(&mut *ZONE.lock().unwrap(), object)
        .unwrap();
    STATIC_HEAP
        .lock()
        .unwrap()
        .free(&mut *ZONE.lock().unwrap(), block)
        .unwrap();

    // Every empty slab goes back to the zone.
    let mut cache = CACHE.lock().unwrap();
    let mut heap = STATIC_HEAP.lock().unwrap();
    let mut zone = ZONE.lock().unwrap();
    cache.shrink(&mut *zone).unwrap();
    assert_eq!(cache.nr_slabs(), 0);
    heap.shrink(&mut *zone).unwrap();
    assert_eq!(zone.buddy().nr_free(), PAGES);

    // The backing allocation stays alive through the zone's mapping.
    core::mem::forget(memory);
}
