//! The SLUB-style slab cache core.
//!
//! The implementation follows `mm/slub.c`:
//!
//! - a slab block is carved into objects on a `stride` grid
//!   (`calculate_sizes`), with the free list pointers stored inside the
//!   free objects (`set_freepointer`/`get_freepointer`);
//! - the cache keeps one active slab (`cpu_slab`) and a doubly linked
//!   partial list (`kmem_cache_node::partial`);
//! - allocation pops from the active slab and refills from the partial
//!   list or from the page allocator (`__slab_alloc` + `new_slab`);
//! - freeing pushes the object back and moves the slab between the full,
//!   partial and empty states (`slab_free`), returning empty slabs to the
//!   page allocator once `min_partial` is exceeded.
//!
//! Deviations from the kernel: the per-CPU arrays are not implemented
//! (locking is external, see [`KmemCache::try_alloc_cached`]), slab
//! metadata lives at the start of the block instead of being overlaid on
//! `struct page`, and cache coloring, poisoning and constructors are
//! future work.
//!
//! TODO(percpu): give each CPU a magazine and a frozen `cpu_slab`, and a
//! lock-free fast path, mirroring `struct per_cpu_pages`-style batching.
//! TODO(poison): implement `SLAB_POISON` and red zones on top of
//! [`KmemCache::free`].
//! TODO(ctor): support constructors and destructors in
//! [`KmemCache::init`] and `new_slab`.
//! TODO(kmalloc): add the size-class facade (`KmemCacheSet`), `kfree`
//! inference and the large-allocation path through the page allocator.

use core::mem::size_of;
use core::ptr::NonNull;

use the_buddy_system::PageFrame;
use the_memblock::PhysAddr;

use crate::error::Error;
use crate::header::SLAB_MAGIC;
use crate::header::SlabHeader;
use crate::layout::SlabLayout;
use crate::pages::PageAlloc;
use crate::pages::PhysMap;

/// Number of objects a slab should hold before the order is grown,
/// mirroring SLUB's `min_objects`.
const MIN_OBJECTS: usize = 4;

/// Number of partial slabs kept before empty ones are returned to the page
/// allocator, mirroring SLUB's default `min_partial`.
const DEFAULT_MIN_PARTIAL: usize = 5;

/// A slab cache, the counterpart of the kernel's `struct kmem_cache`.
///
/// The cache is usable without a global allocator: it is defined in place
/// (a stack local, a `Box`, or a `static`) and initialized once with
/// [`KmemCache::init`], following the same `uninit`/`init` pattern as the
/// buddy allocator.
///
/// The cache stores its own address in every slab header to reject frees
/// through the wrong cache, so it must not move while its slabs are alive;
/// a moved cache fails safely (frees are rejected as
/// [`Error::InvalidPointer`]) but its slabs leak until they are released
/// through the stale address. Kernels define caches in statics, which
/// satisfies the requirement by construction.
#[derive(Debug)]
pub struct KmemCache {
    name: &'static str,
    layout: SlabLayout,
    /// Slab currently receiving allocations (`cpu_slab`).
    current: Option<NonNull<SlabHeader>>,
    /// Head of the doubly linked partial list.
    partial: Option<NonNull<SlabHeader>>,
    nr_partial: usize,
    min_partial: usize,
    nr_slabs: usize,
    nr_objects: usize,
    nr_free_objects: usize,
}

// SAFETY: `KmemCache` holds no thread-affine state, and the slab headers it
// points at are only reached through its methods. Every method's contract
// requires the caller to serialize access (the cache core takes `&mut
// self`), so moving the cache between threads — for example into a spin
// lock — cannot create a data race. `Sync` is deliberately not implemented:
// sharing `&KmemCache` must go through the caller's lock.
unsafe impl Send for KmemCache {}

impl KmemCache {
    /// A `const` placeholder for static definitions.
    ///
    /// The placeholder has no layout; operations that need one report
    /// [`Error::Uninitialized`] until [`KmemCache::init`] runs.
    pub const fn uninit() -> Self {
        Self {
            name: "",
            layout: SlabLayout::EMPTY,
            current: None,
            partial: None,
            nr_partial: 0,
            min_partial: DEFAULT_MIN_PARTIAL,
            nr_slabs: 0,
            nr_objects: 0,
            nr_free_objects: 0,
        }
    }

    /// Returns `true` once [`KmemCache::init`] has computed a layout.
    pub const fn is_initialized(&self) -> bool {
        self.layout.slab_bytes() != 0
    }

    /// Initializes the cache, computing the slab layout from `pages`.
    ///
    /// `size` is the object size in bytes and `align` the requested object
    /// alignment (raised to pointer alignment when smaller, as in SLUB's
    /// `calculate_alignment`). The page size and largest order come from
    /// the page allocator.
    ///
    /// Mirrors `kmem_cache_create` plus the layout part of `new_slab`.
    // TODO(ctor): accept a constructor/destructor callback and run it for
    // every object as slabs are grown or released.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyInitialized`] when called twice,
    /// [`Error::InvalidObjectSize`] for objects smaller than a free list
    /// pointer, [`Error::InvalidAlign`], [`Error::InvalidPageSize`], or
    /// [`Error::ObjectTooLarge`] when no order up to the allocator's
    /// maximum can hold an object.
    pub fn init<PA: PageAlloc>(
        &mut self,
        pages: &PA,
        name: &'static str,
        size: usize,
        align: usize,
    ) -> Result<(), Error> {
        if self.is_initialized() {
            return Err(Error::AlreadyInitialized);
        }

        let layout = SlabLayout::new(
            size_of::<SlabHeader>(),
            size,
            align,
            pages.page_size(),
            pages.max_order(),
            MIN_OBJECTS,
        )?;

        // The block size must be expressible in the address type used to
        // align objects down to their slab.
        let slab_bytes = PA::Addr::from_usize(layout.slab_bytes());
        if slab_bytes.try_to_usize() != Some(layout.slab_bytes()) {
            return Err(Error::ObjectTooLarge);
        }

        self.name = name;
        self.layout = layout;
        self.current = None;
        self.partial = None;
        self.nr_partial = 0;
        self.nr_slabs = 0;
        self.nr_objects = 0;
        self.nr_free_objects = 0;
        Ok(())
    }

    /// Returns the cache name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the layout the cache was initialized with.
    pub const fn layout(&self) -> &SlabLayout {
        &self.layout
    }

    /// Returns the number of slabs handed out by the page allocator.
    pub const fn nr_slabs(&self) -> usize {
        self.nr_slabs
    }

    /// Returns the total number of objects managed by the cache.
    ///
    /// This counter covers full slabs too, which cannot be walked, so it is
    /// not cross-checked by [`KmemCache::validate`].
    pub const fn nr_objects(&self) -> usize {
        self.nr_objects
    }

    /// Returns the number of free objects.
    pub const fn nr_free_objects(&self) -> usize {
        self.nr_free_objects
    }

    /// Sets the number of partial slabs kept before empty ones are
    /// returned to the page allocator (SLUB's `min_partial`).
    pub fn set_min_partial(&mut self, min_partial: usize) {
        self.min_partial = min_partial;
    }

    /// Allocates one object, refilling from the page allocator when the
    /// active slab is full.
    ///
    /// Mirrors `slab_alloc_node`.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KmemCache::init`], or the page
    /// allocator's error when no slab can be grown.
    pub fn alloc<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<NonNull<u8>, Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }
        if let Some(object) = self.try_alloc_cached() {
            return Ok(object);
        }
        self.refill(pages)?;
        self.try_alloc_cached().ok_or(Error::InternalError)
    }

    /// Allocates one object with the object bytes zeroed.
    pub fn alloc_zeroed<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<NonNull<u8>, Error> {
        let object = self.alloc(pages)?;
        // SAFETY: the object holds `object_size` writable bytes.
        unsafe { object.as_ptr().write_bytes(0, self.layout.object_size()) };
        Ok(object)
    }

    /// Allocates from the active slab only, without touching the page
    /// allocator.
    ///
    /// This is the fast path a kernel would run without the zone lock;
    /// with an external lock it keeps the lock hold time short. Returns
    /// `None` when the active slab has no free objects.
    // TODO(percpu): run this against a per-CPU active slab so it needs no
    // lock at all, mirroring SLUB's `cpu_slab`.
    pub fn try_alloc_cached(&mut self) -> Option<NonNull<u8>> {
        let current = self.current?;

        // SAFETY: `current` is a slab owned by this cache, and `&mut self`
        // guarantees exclusive access to it.
        unsafe {
            let header = &mut *current.as_ptr();
            let object = header.freelist?;
            debug_assert!(header.inuse < header.objects);

            header.freelist = get_freepointer(object);
            header.inuse += 1;
            debug_assert!(self.nr_free_objects > 0);
            self.nr_free_objects -= 1;
            Some(object)
        }
    }

    /// Frees an object previously returned by [`KmemCache::alloc`].
    ///
    /// Mirrors `slab_free`: the object goes back to its slab's free list
    /// and the slab moves between the full, partial and empty states. An
    /// empty, non-active slab is returned to the page allocator.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`] before [`KmemCache::init`];
    /// [`Error::InvalidPointer`] if the pointer is not an object of this
    /// cache (misaligned, outside the object area, or not a slab);
    /// [`Error::CrossCacheFree`] if the slab belongs to another cache;
    /// [`Error::DoubleFree`] if the object is already free;
    /// [`Error::PageAlloc`] if returning an empty slab fails.
    pub fn free<PA: PageAlloc>(&mut self, pages: &mut PA, ptr: NonNull<u8>) -> Result<(), Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }

        let owner = (self as *const KmemCache).cast::<()>();
        let phys = pages.virt_to_phys(ptr);
        let slab_bytes = PA::Addr::from_usize(self.layout.slab_bytes());
        let base = PA::Addr::align_down(phys, slab_bytes);
        let slab = pages.phys_to_virt(base).cast::<SlabHeader>();

        // SAFETY: `base` is block aligned, so the header resides at a
        // valid, aligned location. It is either a live slab of some cache
        // or arbitrary (initialized) memory; the magic and owner checks
        // reject the latter before anything is mutated.
        let (inuse, objects, on_partial) = unsafe {
            let header = &mut *slab.as_ptr();
            if header.magic != SLAB_MAGIC {
                return Err(Error::InvalidPointer);
            }
            if header.owner != owner {
                return Err(Error::CrossCacheFree);
            }
            if header.order != self.layout.order() {
                return Err(Error::InvalidPointer);
            }

            // The object must lie on the stride grid of the object area.
            let offset = (phys - base).try_to_usize().ok_or(Error::InvalidPointer)?;
            let first = self.layout.first_offset();
            let stride = self.layout.stride();
            if offset < first
                || offset >= self.layout.slab_bytes()
                || (offset - first) % stride != 0
            {
                return Err(Error::InvalidPointer);
            }

            // Best-effort double free detection: the free list head catches
            // the common free-then-free case, the in-use count catches
            // frees once the slab has nothing allocated.
            if header.freelist == Some(ptr) || header.inuse == 0 {
                return Err(Error::DoubleFree);
            }

            // TODO(poison): write the poison pattern here and check it on
            // allocation and in `validate`.
            set_freepointer(ptr, header.freelist);
            header.freelist = Some(ptr);
            header.inuse -= 1;
            (header.inuse, header.objects, header.on_partial)
        };
        self.nr_free_objects += 1;

        if inuse == 0 {
            if self.current == Some(slab) {
                // Kept as the active slab for fast reuse.
            } else if on_partial {
                if self.nr_partial > self.min_partial {
                    self.unlink_partial(slab);
                    self.release_slab(pages, slab)?;
                }
            } else {
                // The slab was full and is now empty; nothing links to it.
                self.release_slab(pages, slab)?;
            }
        } else if inuse < objects && !on_partial && self.current != Some(slab) {
            // The slab was full and is now partial.
            self.push_partial(slab);
        }

        Ok(())
    }

    /// Returns every empty slab to the page allocator.
    ///
    /// Mirrors `kmem_cache_shrink`: partial slabs with live objects stay
    /// cached, empty ones (including the active slab) are released. All
    /// objects must be freed for the cache to release everything.
    pub fn shrink<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<(), Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }

        if let Some(current) = self.current {
            // SAFETY: `current` is owned by this cache.
            let inuse = unsafe { (*current.as_ptr()).inuse };
            if inuse == 0 {
                self.current = None;
                self.release_slab(pages, current)?;
            }
        }

        let mut node = self.partial;
        while let Some(slab) = node {
            // SAFETY: partial list nodes are owned slabs.
            let (next, inuse) = unsafe {
                let header = &*slab.as_ptr();
                (header.next, header.inuse)
            };
            node = next;
            if inuse == 0 {
                self.unlink_partial(slab);
                self.release_slab(pages, slab)?;
            }
        }

        Ok(())
    }

    /// Releases every slab and consumes the cache.
    ///
    /// Mirrors `kmem_cache_destroy`. All objects must have been freed
    /// first; slabs holding live objects are not reachable through the
    /// partial list and keep their pages.
    pub fn destroy<PA: PageAlloc>(mut self, pages: &mut PA) -> Result<(), Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }

        if let Some(current) = self.current.take() {
            self.release_slab(pages, current)?;
        }
        while let Some(slab) = self.partial {
            self.unlink_partial(slab);
            self.release_slab(pages, slab)?;
        }

        Ok(())
    }

    /// Verifies the cache's invariants.
    ///
    /// Checks every walkable slab (the active one and the partial list):
    /// header magic, owner and layout, in-use accounting, the object free
    /// list (bounded, aligned, inside the object area) and the partial
    /// list links. Full slabs cannot be enumerated, exactly as in SLUB, so
    /// only the free object count is cross-checked against the
    /// bookkeeping.
    ///
    /// Mirrors `check_object` and `validate_slab_cache`.
    ///
    /// # Errors
    ///
    /// [`Error::Uninitialized`], [`Error::CorruptSlab`] or
    /// [`Error::CountMismatch`] for the first violated invariant.
    pub fn validate<M: PhysMap>(&self, map: &M) -> Result<(), Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }

        let mut free_total = 0usize;

        if let Some(current) = self.current {
            // SAFETY: `current` is owned by this cache.
            let on_partial = unsafe { (*current.as_ptr()).on_partial };
            if on_partial {
                return Err(Error::CorruptSlab);
            }
            free_total += self.validate_slab(map, current)?;
        }

        let mut node = self.partial;
        let mut len = 0usize;
        let mut prev: Option<NonNull<SlabHeader>> = None;
        while let Some(slab) = node {
            // SAFETY: partial list nodes are owned slabs.
            let (next, on_partial, back) = unsafe {
                let header = &*slab.as_ptr();
                (header.next, header.on_partial, header.prev)
            };
            if !on_partial || back != prev {
                return Err(Error::CorruptSlab);
            }
            free_total += self.validate_slab(map, slab)?;
            len += 1;
            if len > self.nr_slabs {
                return Err(Error::CorruptSlab);
            }
            prev = node;
            node = next;
        }

        if len != self.nr_partial {
            return Err(Error::CountMismatch);
        }
        if free_total != self.nr_free_objects {
            return Err(Error::CountMismatch);
        }

        Ok(())
    }

    /// Verifies one slab, returning its number of free objects.
    fn validate_slab<M: PhysMap>(
        &self,
        map: &M,
        slab: NonNull<SlabHeader>,
    ) -> Result<usize, Error> {
        // SAFETY: the caller guarantees `slab` is owned by this cache.
        unsafe {
            let header = &*slab.as_ptr();
            let owner = (self as *const KmemCache).cast::<()>();
            if header.magic != SLAB_MAGIC || header.owner != owner {
                return Err(Error::CorruptSlab);
            }
            if header.order != self.layout.order() || header.objects != self.layout.objects() {
                return Err(Error::CorruptSlab);
            }
            if header.inuse > header.objects {
                return Err(Error::CountMismatch);
            }

            let base = slab.cast::<u8>();
            let base_phys = map.virt_to_phys(base);
            let mut node = header.freelist;
            let mut free = 0usize;
            while let Some(object) = node {
                free += 1;
                if free > header.objects {
                    return Err(Error::CorruptSlab);
                }

                let phys = map.virt_to_phys(object);
                if phys < base_phys {
                    return Err(Error::CorruptSlab);
                }
                let offset = (phys - base_phys)
                    .try_to_usize()
                    .ok_or(Error::CorruptSlab)?;
                if offset < self.layout.first_offset()
                    || offset + self.layout.object_size() > self.layout.slab_bytes()
                    || (offset - self.layout.first_offset()) % self.layout.stride() != 0
                {
                    return Err(Error::CorruptSlab);
                }

                node = get_freepointer(object);
            }

            if free != header.objects - header.inuse {
                return Err(Error::CountMismatch);
            }
            Ok(free)
        }
    }

    /// Refills the active slab from the partial list or the page allocator.
    fn refill<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<(), Error> {
        self.shelve_current(pages)?;

        if let Some(slab) = self.pop_partial() {
            self.current = Some(slab);
            return Ok(());
        }

        self.current = Some(self.new_slab(pages)?);
        Ok(())
    }

    /// Moves the active slab aside when it has no free objects.
    fn shelve_current<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<(), Error> {
        let Some(current) = self.current.take() else {
            return Ok(());
        };

        // SAFETY: `current` is owned by this cache.
        let (inuse, objects) = unsafe {
            let header = &*current.as_ptr();
            (header.inuse, header.objects)
        };

        if inuse == 0 {
            // Empty slabs are kept within `min_partial` and released past
            // it, mirroring SLUB's empty slab policy.
            if self.nr_partial >= self.min_partial {
                self.release_slab(pages, current)?;
            } else {
                self.push_partial(current);
            }
        } else if inuse < objects {
            self.push_partial(current);
        }
        // A full slab is forgotten: `free` finds it through its objects.
        Ok(())
    }

    /// Grows a new slab from the page allocator and publishes it as the
    /// active slab.
    ///
    /// Mirrors `allocate_slab`: the header is filled in and the object
    /// free list is built before the slab can be reached.
    fn new_slab<PA: PageAlloc>(&mut self, pages: &mut PA) -> Result<NonNull<SlabHeader>, Error> {
        let block = pages.alloc_pages(self.layout.order())?;
        let base = pages.phys_to_virt(block);
        let slab = base.cast::<SlabHeader>();

        // SAFETY: the page allocator returned an exclusive, block-aligned
        // block that is large enough for the header and every object (the
        // layout guarantees `first_offset + objects * stride <=
        // slab_bytes`).
        unsafe {
            let header = &mut *slab.as_ptr();
            header.owner = (self as *const KmemCache).cast::<()>();
            header.magic = SLAB_MAGIC;
            header.order = self.layout.order();
            header.on_partial = false;
            header.inuse = 0;
            header.objects = self.layout.objects();
            header.freelist = None;
            header.next = None;
            header.prev = None;

            for index in (0..self.layout.objects()).rev() {
                let object = base.add(self.layout.first_offset() + index * self.layout.stride());
                set_freepointer(object, header.freelist);
                header.freelist = Some(object);
            }
        }

        self.nr_slabs += 1;
        self.nr_objects += self.layout.objects();
        self.nr_free_objects += self.layout.objects();
        Ok(slab)
    }

    /// Returns a slab to the page allocator.
    fn release_slab<PA: PageAlloc>(
        &mut self,
        pages: &mut PA,
        slab: NonNull<SlabHeader>,
    ) -> Result<(), Error> {
        // SAFETY: `slab` is owned by this cache and linked nowhere.
        let (inuse, objects) = unsafe {
            let header = &mut *slab.as_ptr();
            header.magic = 0;
            (header.inuse, header.objects)
        };

        let phys = pages.virt_to_phys(slab.cast());
        pages.free_pages(phys, self.layout.order())?;

        self.nr_slabs -= 1;
        self.nr_objects -= objects;
        self.nr_free_objects -= objects - inuse;
        Ok(())
    }

    /// Pushes a slab at the front of the partial list.
    fn push_partial(&mut self, slab: NonNull<SlabHeader>) {
        // SAFETY: `slab` is owned by this cache and not linked anywhere.
        unsafe {
            debug_assert!(!(*slab.as_ptr()).on_partial);
            let header = &mut *slab.as_ptr();
            header.on_partial = true;
            header.prev = None;
            header.next = self.partial;
            if let Some(head) = self.partial {
                (*head.as_ptr()).prev = Some(slab);
            }
        }
        self.partial = Some(slab);
        self.nr_partial += 1;
    }

    /// Pops the front slab of the partial list.
    fn pop_partial(&mut self) -> Option<NonNull<SlabHeader>> {
        let slab = self.partial?;

        // SAFETY: `slab` is the head of the partial list.
        unsafe {
            let header = &mut *slab.as_ptr();
            self.partial = header.next;
            header.on_partial = false;
            header.next = None;
            header.prev = None;
            if let Some(next) = self.partial {
                (*next.as_ptr()).prev = None;
            }
        }
        self.nr_partial -= 1;
        Some(slab)
    }

    /// Unlinks a slab from the partial list.
    fn unlink_partial(&mut self, slab: NonNull<SlabHeader>) {
        // SAFETY: `slab` is linked into the partial list.
        unsafe {
            let header = &mut *slab.as_ptr();
            debug_assert!(header.on_partial);

            match (header.prev, header.next) {
                (None, None) => self.partial = None,
                (None, Some(next)) => {
                    (*next.as_ptr()).prev = None;
                    self.partial = Some(next);
                }
                (Some(prev), None) => {
                    (*prev.as_ptr()).next = None;
                }
                (Some(prev), Some(next)) => {
                    (*prev.as_ptr()).next = Some(next);
                    (*next.as_ptr()).prev = Some(prev);
                }
            }

            header.on_partial = false;
            header.next = None;
            header.prev = None;
        }
        self.nr_partial -= 1;
    }
}

/// Writes the free list pointer into a free object (`set_freepointer`).
///
/// # Safety
///
/// `object` must be an object of the calling cache, which guarantees at
/// least pointer-sized, pointer-aligned storage.
unsafe fn set_freepointer(object: NonNull<u8>, next: Option<NonNull<u8>>) {
    let next = next.map_or(core::ptr::null_mut(), |ptr| ptr.as_ptr());
    // SAFETY: guaranteed by the caller.
    unsafe { object.cast::<*mut u8>().write(next) };
}

/// Reads the free list pointer from a free object (`get_freepointer`).
///
/// # Safety
///
/// Same contract as [`set_freepointer`].
unsafe fn get_freepointer(object: NonNull<u8>) -> Option<NonNull<u8>> {
    // SAFETY: guaranteed by the caller.
    let next = unsafe { object.cast::<*mut u8>().read() };
    NonNull::new(next)
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;
    use core::alloc::Layout;

    use super::*;
    use crate::pages::DirectMap;

    const PAGE: usize = 4096;
    const MAX_ORDER: usize = 3;
    const PAGES: usize = 64;

    /// A page allocator over a real, block-aligned allocation with an
    /// identity mapping; stands in for the buddy allocator.
    struct TestAlloc {
        base: NonNull<u8>,
        memory_layout: Layout,
        map: DirectMap<usize>,
        max_order: u8,
        /// Order of the block starting at each page, `None` when free.
        blocks: Vec<Option<u8>>,
    }

    impl TestAlloc {
        fn with_max_order(max_order: u8) -> Self {
            let block_bytes = PAGE << max_order;
            let memory_layout =
                Layout::from_size_align(PAGES * PAGE, block_bytes).expect("valid layout");
            // SAFETY: `memory_layout` has a non-zero size.
            let base = unsafe { alloc::alloc::alloc(memory_layout) };
            let base = NonNull::new(base).expect("allocation failed");
            // SAFETY: the allocation spans `PAGES * PAGE` writable bytes;
            // zeroing makes header reads on bogus pointers deterministic
            // (and initialized, as Miri requires).
            unsafe { core::ptr::write_bytes(base.as_ptr(), 0, PAGES * PAGE) };
            // SAFETY: the identity mapping is anchored at the allocation.
            let map = unsafe { DirectMap::new(base.as_ptr().addr(), base) };

            Self {
                base,
                memory_layout,
                map,
                max_order,
                blocks: vec![None; PAGES],
            }
        }

        fn new() -> Self {
            Self::with_max_order(MAX_ORDER as u8)
        }

        fn ptr_at(&self, page: usize) -> NonNull<u8> {
            debug_assert!(page < PAGES);
            // SAFETY: `page < PAGES` keeps the arithmetic inside the
            // allocation.
            unsafe { self.base.add(page * PAGE) }
        }

        fn page_of(&self, ptr: NonNull<u8>) -> usize {
            (ptr.as_ptr().addr() - self.base.as_ptr().addr()) / PAGE
        }

        fn allocated_pages(&self) -> usize {
            self.blocks.iter().filter(|block| block.is_some()).count()
        }

        fn find_run(&self, order: u8) -> Option<usize> {
            let run = 1usize << order;
            let mut start = 0;
            while start + run <= self.blocks.len() {
                if self.blocks[start..start + run].iter().all(|b| b.is_none()) {
                    return Some(start);
                }
                start += run;
            }
            None
        }
    }

    impl Drop for TestAlloc {
        fn drop(&mut self) {
            // SAFETY: the allocation came from `alloc` with this layout.
            unsafe { alloc::alloc::dealloc(self.base.as_ptr(), self.memory_layout) };
        }
    }

    impl PhysMap for TestAlloc {
        type Addr = usize;

        fn phys_to_virt(&self, addr: usize) -> NonNull<u8> {
            self.map.phys_to_virt(addr)
        }

        fn virt_to_phys(&self, ptr: NonNull<u8>) -> usize {
            self.map.virt_to_phys(ptr)
        }
    }

    impl PageAlloc for TestAlloc {
        fn page_size(&self) -> usize {
            PAGE
        }

        fn max_order(&self) -> u8 {
            self.max_order
        }

        fn alloc_pages(&mut self, order: u8) -> Result<usize, Error> {
            let start = self.find_run(order).ok_or(Error::OutOfMemory)?;
            for block in &mut self.blocks[start..start + (1 << order)] {
                *block = Some(order);
            }
            Ok(self.base.as_ptr().addr() + start * PAGE)
        }

        fn free_pages(&mut self, addr: usize, order: u8) -> Result<(), Error> {
            let start = (addr - self.base.as_ptr().addr()) / PAGE;
            let run = 1usize << order;
            if start + run > self.blocks.len()
                || self.blocks[start..start + run]
                    .iter()
                    .any(|block| *block != Some(order))
            {
                return Err(Error::PageAlloc);
            }
            for block in &mut self.blocks[start..start + run] {
                *block = None;
            }
            Ok(())
        }
    }

    fn cache_with(alloc: &TestAlloc, size: usize, align: usize) -> KmemCache {
        let mut cache = KmemCache::uninit();
        cache.init(alloc, "test", size, align).unwrap();
        cache
    }

    fn alloc_many(
        cache: &mut KmemCache,
        alloc: &mut TestAlloc,
        count: usize,
    ) -> Vec<NonNull<u8>> {
        (0..count).map(|_| cache.alloc(alloc).unwrap()).collect()
    }

    #[test]
    fn uninit_is_inert() {
        let mut alloc = TestAlloc::new();
        let mut cache = KmemCache::uninit();
        assert!(!cache.is_initialized());
        assert_eq!(cache.nr_slabs(), 0);

        assert!(matches!(cache.alloc(&mut alloc), Err(Error::Uninitialized)));
        assert!(matches!(
            cache.free(&mut alloc, NonNull::dangling()),
            Err(Error::Uninitialized)
        ));
        assert!(matches!(cache.shrink(&mut alloc), Err(Error::Uninitialized)));
        assert!(matches!(cache.validate(&alloc), Err(Error::Uninitialized)));
        assert!(matches!(
            cache.destroy(&mut alloc),
            Err(Error::Uninitialized)
        ));
    }

    #[test]
    fn init_validates_parameters_and_runs_once() {
        let alloc = TestAlloc::new();
        let mut cache = KmemCache::uninit();

        assert!(matches!(
            cache.init(&alloc, "tiny", 4, 8),
            Err(Error::InvalidObjectSize)
        ));
        assert!(matches!(
            cache.init(&alloc, "bad align", 32, 3),
            Err(Error::InvalidAlign)
        ));
        assert!(matches!(
            cache.init(&alloc, "too large", 1 << 20, 8),
            Err(Error::ObjectTooLarge)
        ));
        assert!(!cache.is_initialized());

        cache.init(&alloc, "objects", 24, 8).unwrap();
        assert!(cache.is_initialized());
        assert_eq!(cache.name(), "objects");
        assert_eq!(cache.layout().object_size(), 24);
        assert_eq!(cache.layout().objects(), (PAGE - 56) / 24);
        assert!(matches!(
            cache.init(&alloc, "again", 24, 8),
            Err(Error::AlreadyInitialized)
        ));
    }

    #[test]
    fn objects_fill_a_slab_before_a_new_one_is_grown() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        let per_slab = cache.layout().objects();

        let objects = alloc_many(&mut cache, &mut alloc, per_slab);
        assert_eq!(cache.nr_slabs(), 1);
        assert_eq!(cache.nr_objects(), per_slab);
        assert_eq!(cache.nr_free_objects(), 0);

        let first_page = alloc.page_of(objects[0]);
        for object in &objects {
            assert_eq!(object.as_ptr().addr() % 8, 0);
            assert_eq!(alloc.page_of(*object), first_page);
        }
        let mut addresses: Vec<usize> = objects.iter().map(|o| o.as_ptr().addr()).collect();
        addresses.sort_unstable();
        addresses.dedup();
        assert_eq!(addresses.len(), objects.len());

        // One more object spills into a second slab.
        let extra = cache.alloc(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 2);
        assert_eq!(cache.nr_objects(), 2 * per_slab);
        assert_eq!(alloc.page_of(extra), first_page + 1);
        cache.validate(&alloc).unwrap();

        for object in objects {
            cache.free(&mut alloc, object).unwrap();
        }
        cache.free(&mut alloc, extra).unwrap();
        assert_eq!(cache.nr_free_objects(), 2 * per_slab);
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn cached_fast_path_drains_and_refills() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 64, 8);

        // Grow the first slab through `alloc`, then drain it through the
        // fast path alone.
        let first = cache.alloc(&mut alloc).unwrap();
        let mut drained = 1;
        while cache.try_alloc_cached().is_some() {
            drained += 1;
        }
        assert_eq!(drained, cache.layout().objects());
        assert_eq!(cache.nr_free_objects(), 0);

        // The full path grows a second slab.
        let second = cache.alloc(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 2);
        assert_ne!(alloc.page_of(first), alloc.page_of(second));
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn free_rejects_bad_pointers() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        let object = cache.alloc(&mut alloc).unwrap();

        // Misaligned pointer inside the object area.
        let misaligned = NonNull::new(object.as_ptr().wrapping_add(1)).unwrap();
        assert!(matches!(
            cache.free(&mut alloc, misaligned),
            Err(Error::InvalidPointer)
        ));

        // Pointer into the header area of a valid slab.
        let header_slot = unsafe { alloc.ptr_at(alloc.page_of(object)).add(8) };
        assert!(matches!(
            cache.free(&mut alloc, header_slot),
            Err(Error::InvalidPointer)
        ));

        // Pointer into memory that is not a slab: zeroed memory fails the
        // magic check.
        let stray = alloc.ptr_at(32);
        assert!(matches!(
            cache.free(&mut alloc, stray),
            Err(Error::InvalidPointer)
        ));

        // A pointer of another cache with an identical layout.
        let mut other = cache_with(&alloc, 24, 8);
        let other_object = other.alloc(&mut alloc).unwrap();
        assert!(matches!(
            cache.free(&mut alloc, other_object),
            Err(Error::CrossCacheFree)
        ));
        other.free(&mut alloc, other_object).unwrap();

        // Immediate double free.
        cache.free(&mut alloc, object).unwrap();
        assert!(matches!(
            cache.free(&mut alloc, object),
            Err(Error::DoubleFree)
        ));
    }

    #[test]
    fn partial_slabs_are_reused() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        let per_slab = cache.layout().objects();

        // Fill the first slab and grow a second one.
        let first_objects = alloc_many(&mut cache, &mut alloc, per_slab);
        let page_a = alloc.page_of(first_objects[0]);
        let second = cache.alloc(&mut alloc).unwrap();
        assert_eq!(alloc.page_of(second), page_a + 1);
        assert_eq!(cache.nr_slabs(), 2);

        // Free one object of the forgotten, full first slab: it becomes
        // partial and is linked into the partial list.
        cache.free(&mut alloc, first_objects[0]).unwrap();

        // Exhaust the second slab; the refill must reuse slab A.
        for _ in 1..per_slab {
            cache.alloc(&mut alloc).unwrap();
        }
        let reused = cache.alloc(&mut alloc).unwrap();
        assert_eq!(alloc.page_of(reused), page_a);
        assert_eq!(cache.nr_slabs(), 2);
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn empty_slabs_are_released_to_the_page_allocator() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        let per_slab = cache.layout().objects();

        let objects = alloc_many(&mut cache, &mut alloc, 2 * per_slab);
        assert_eq!(cache.nr_slabs(), 2);
        assert_eq!(alloc.allocated_pages(), 2);

        for object in objects {
            cache.free(&mut alloc, object).unwrap();
        }
        // Both slabs are empty: the active one is kept, the other is kept
        // on the partial list within `min_partial`.
        assert_eq!(cache.nr_slabs(), 2);
        assert_eq!(alloc.allocated_pages(), 2);
        assert_eq!(cache.nr_free_objects(), 2 * per_slab);
        cache.validate(&alloc).unwrap();

        // Shrinking releases every empty slab.
        cache.shrink(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 0);
        assert_eq!(alloc.allocated_pages(), 0);
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn min_partial_bounds_cached_empty_slabs() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        cache.set_min_partial(0);
        let per_slab = cache.layout().objects();

        let objects = alloc_many(&mut cache, &mut alloc, 2 * per_slab);
        for object in objects {
            cache.free(&mut alloc, object).unwrap();
        }

        // With `min_partial` zero the emptied partial slab goes back to
        // the page allocator; only the active slab remains.
        assert_eq!(cache.nr_slabs(), 1);
        assert_eq!(alloc.allocated_pages(), 1);
        cache.shrink(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 0);
        assert_eq!(alloc.allocated_pages(), 0);
    }

    #[test]
    fn single_object_slabs_are_released_when_empty() {
        let mut alloc = TestAlloc::with_max_order(0);
        let mut cache = cache_with(&alloc, 2048, 8);
        assert_eq!(cache.layout().objects(), 1);

        // Fill the first single-object slab, then grow a second one.
        let first = cache.alloc(&mut alloc).unwrap();
        let second = cache.alloc(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 2);
        assert_ne!(alloc.page_of(first), alloc.page_of(second));

        // The first slab was full, is now empty and linked nowhere: it is
        // released immediately, together with the object just freed.
        cache.free(&mut alloc, first).unwrap();
        assert_eq!(cache.nr_slabs(), 1);
        assert_eq!(alloc.allocated_pages(), 1);
        assert_eq!(cache.nr_objects(), 1);
        assert_eq!(cache.nr_free_objects(), 0);
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn alloc_zeroed_clears_the_object() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);

        let first = cache.alloc(&mut alloc).unwrap();
        // SAFETY: the object is allocated and writable.
        unsafe { first.as_ptr().write_bytes(0xaa, 24) };

        let zeroed = cache.alloc_zeroed(&mut alloc).unwrap();
        // SAFETY: the object is allocated and readable.
        let bytes = unsafe { core::slice::from_raw_parts(zeroed.as_ptr(), 24) };
        assert!(bytes.iter().all(|byte| *byte == 0));

        cache.free(&mut alloc, first).unwrap();
        cache.free(&mut alloc, zeroed).unwrap();
        cache.validate(&alloc).unwrap();
    }

    #[test]
    fn destroy_returns_every_slab() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 512, 8);

        let count = cache.layout().objects() * 3;
        let objects = alloc_many(&mut cache, &mut alloc, count);
        assert_eq!(cache.nr_slabs(), 3);
        for object in objects {
            cache.free(&mut alloc, object).unwrap();
        }

        cache.destroy(&mut alloc).unwrap();
        assert_eq!(alloc.allocated_pages(), 0);
    }
    #[test]
    fn validate_detects_corruption() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 24, 8);
        let object = cache.alloc(&mut alloc).unwrap();
        let slab = alloc.ptr_at(alloc.page_of(object)).cast::<SlabHeader>();

        // Writes go through the raw pointer, because every `validate` call
        // derives shared references from it, which a live `&mut` would keep
        // invalidated (Stacked Borrows).
        // SAFETY: the slab header is owned by this cache.
        unsafe {
            let magic = (*slab.as_ptr()).magic;

            (*slab.as_ptr()).magic = 0;
            assert_eq!(cache.validate(&alloc).unwrap_err(), Error::CorruptSlab);

            (*slab.as_ptr()).magic = magic;
            (*slab.as_ptr()).inuse = (*slab.as_ptr()).objects + 1;
            assert_eq!(cache.validate(&alloc).unwrap_err(), Error::CountMismatch);

            (*slab.as_ptr()).inuse = 0;
            let freelist = (*slab.as_ptr()).freelist;
            // Create a cycle in the free list.
            let first = slab.cast::<u8>().add(cache.layout().first_offset());
            set_freepointer(first, Some(first));
            (*slab.as_ptr()).freelist = Some(first);
            assert_eq!(cache.validate(&alloc).unwrap_err(), Error::CorruptSlab);

            (*slab.as_ptr()).freelist = freelist;
        }
    }

    #[test]
    fn object_alignment_is_honored() {
        let mut alloc = TestAlloc::new();
        for align in [8usize, 16, 32, 64, 128] {
            let mut cache = cache_with(&alloc, 80, align);
            let count = cache.layout().objects();
            let objects = alloc_many(&mut cache, &mut alloc, count);
            for object in &objects {
                assert_eq!(
                    object.as_ptr().addr() % align,
                    0,
                    "alignment {align} violated"
                );
            }
            for object in objects {
                cache.free(&mut alloc, object).unwrap();
            }
            cache.shrink(&mut alloc).unwrap();
            assert_eq!(cache.nr_slabs(), 0);
        }
        assert_eq!(alloc.allocated_pages(), 0);
    }

    #[test]
    fn random_operations_keep_invariants() {
        let mut alloc = TestAlloc::new();
        let mut cache = cache_with(&alloc, 48, 8);

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut live: Vec<NonNull<u8>> = Vec::new();
        for round in 0..4000 {
            if rand() % 3 != 0 && !live.is_empty() {
                let index = (rand() as usize) % live.len();
                let object = live.swap_remove(index);
                cache.free(&mut alloc, object).unwrap();
            } else if let Ok(object) = cache.alloc(&mut alloc) {
                live.push(object);
            }
            if round % 100 == 0 {
                cache.validate(&alloc).unwrap();
            }
        }

        for object in live {
            cache.free(&mut alloc, object).unwrap();
        }
        cache.validate(&alloc).unwrap();
        cache.shrink(&mut alloc).unwrap();
        assert_eq!(cache.nr_slabs(), 0);
        assert_eq!(alloc.allocated_pages(), 0);
        cache.validate(&alloc).unwrap();
    }
}
