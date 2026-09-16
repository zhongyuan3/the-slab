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
use the_slab::KmemCache;

const PAGE: usize = 0x1000;
const PAGES: usize = 64;
const MAX_ORDER: usize = 4; // largest block: 8 pages
const BLOCK_BYTES: usize = PAGE << (MAX_ORDER - 1);

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
    let mut buddy = Buddy::<usize, MAX_ORDER>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_range(base, base + PAGES * PAGE).unwrap();
    assert_eq!(buddy.nr_free(), PAGES);

    // SAFETY: the identity mapping covers the backing allocation, which
    // outlives the page allocator and every cache using it.
    let map = unsafe { DirectMap::new(base, memory.ptr) };
    let mut pages = BuddyPages::new(&mut buddy, map);

    let mut cache = KmemCache::uninit();
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
    let mut buddy = Buddy::<usize, MAX_ORDER>::new(base, PAGE, &mut descriptors).unwrap();
    buddy.free_memblock(&mb).unwrap();

    {
        // SAFETY: the identity mapping covers the backing allocation.
        let map = unsafe { DirectMap::new(base, memory.ptr) };
        let mut pages = BuddyPages::new(&mut buddy, map);

        let mut cache = KmemCache::uninit();
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
static CACHE: Mutex<KmemCache> = Mutex::new(KmemCache::uninit());

#[test]
fn cache_can_be_defined_in_a_static() {
    let guard = CACHE.lock().unwrap();
    assert!(!guard.is_initialized());
    assert_eq!(guard.nr_slabs(), 0);
}
