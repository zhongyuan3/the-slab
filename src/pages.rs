//! The page allocator and address mapping abstractions.
//!
//! A slab cache never touches the buddy allocator directly: it asks a
//! [`PageAlloc`] for naturally aligned blocks of `2^order` pages and maps
//! the returned addresses through a [`PhysMap`]. This mirrors the kernel's
//! split between the slab allocator and the page allocator (`new_slab` →
//! `alloc_pages_node`) and keeps physical-address handling out of the slab
//! core.
//!
//! Two adapters implement [`PageAlloc`] over the buddy allocator:
//!
//! - [`BuddyPages`] borrows the buddy allocator and a mapping; it suits
//!   one-shot or scoped use, such as tests;
//! - [`Zone`] owns both by value and can therefore live in a `static`,
//!   which is how a kernel keeps its page allocator: caches and the kernel
//!   heap hold no reference to it and receive a mutable borrow per call.

use core::ptr::NonNull;

use the_buddy_system::Buddy;
use the_buddy_system::Error as BuddyError;
use the_buddy_system::Page;
use the_buddy_system::PageFrame;

use crate::error::Error;

/// Translates between physical addresses and the address space the
/// allocator accesses memory through.
///
/// A kernel implements this once for its direct map; tests can implement
/// it with real pointers so that pointer provenance is preserved.
pub trait PhysMap {
    /// The physical address type of the paired page allocator.
    type Addr: PageFrame;

    /// Returns the address at which `addr` is readable and writable.
    ///
    /// The mapping must preserve byte offsets inside each page block,
    /// because a slab lays its objects out physically.
    fn phys_to_virt(&self, addr: Self::Addr) -> NonNull<u8>;

    /// Inverse of [`PhysMap::phys_to_virt`].
    fn virt_to_phys(&self, ptr: NonNull<u8>) -> Self::Addr;
}

/// A page allocator handing out naturally aligned blocks of `2^order`
/// pages.
///
/// Mirrors the page allocator interface SLUB uses (`alloc_pages_node` and
/// `__free_pages`), without `GFP` flags or zone selection.
pub trait PageAlloc: PhysMap {
    /// Returns the page size in bytes.
    fn page_size(&self) -> usize;

    /// Returns the largest order [`PageAlloc::alloc_pages`] accepts.
    fn max_order(&self) -> u8;

    /// Allocates `2^order` contiguous, block-aligned pages.
    fn alloc_pages(&mut self, order: u8) -> Result<Self::Addr, Error>;

    /// Returns a block previously handed out by
    /// [`PageAlloc::alloc_pages`] with the same order.
    fn free_pages(&mut self, addr: Self::Addr, order: u8) -> Result<(), Error>;
}

/// A linear mapping of the form `virt = virt_base + (phys - phys_base)`.
///
/// This is the shape of a kernel's direct map (`PAGE_OFFSET`). The
/// pointer arithmetic is anchored at `virt_base`, so the resulting
/// pointers keep their provenance instead of being forged from integers.
///
/// A mapping can be defined in a `static` with [`DirectMap::uninit`] and
/// filled in at boot with [`DirectMap::init`].
#[derive(Clone, Copy, Debug)]
pub struct DirectMap<A: PageFrame> {
    phys_base: A,
    virt_base: NonNull<u8>,
    initialized: bool,
}

impl<A: PageFrame> DirectMap<A> {
    /// A `const` placeholder for static definitions.
    ///
    /// The placeholder must not be used for address translation before
    /// [`DirectMap::init`] runs; the translation methods panic otherwise.
    pub const fn uninit() -> Self {
        Self {
            phys_base: A::ZERO,
            virt_base: NonNull::dangling(),
            initialized: false,
        }
    }

    /// Creates a mapping.
    ///
    /// # Safety
    ///
    /// - `virt_base` must be the address at which the physical byte
    ///   `phys_base` is mapped for reads and writes;
    /// - the mapped range must cover every block the paired page allocator
    ///   hands out, and must be reserved for the slab allocator's use while
    ///   a cache using this mapping is alive.
    pub unsafe fn new(phys_base: A, virt_base: NonNull<u8>) -> Self {
        Self {
            phys_base,
            virt_base,
            initialized: true,
        }
    }

    /// Initializes a placeholder created by [`DirectMap::uninit`].
    ///
    /// # Safety
    ///
    /// Same contract as [`DirectMap::new`].
    pub unsafe fn init(&mut self, phys_base: A, virt_base: NonNull<u8>) {
        // SAFETY: the caller guarantees the mapping contract.
        *self = unsafe { Self::new(phys_base, virt_base) };
    }

    /// Returns `true` once a mapping has been supplied.
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the physical address the mapping starts at.
    pub fn phys_base(&self) -> A {
        self.phys_base
    }
}

impl<A: PageFrame> PhysMap for DirectMap<A> {
    type Addr = A;

    fn phys_to_virt(&self, addr: A) -> NonNull<u8> {
        assert!(self.initialized, "the physical mapping is not initialized");
        if addr < self.phys_base {
            panic!("address below the mapped base");
        }
        let offset = (addr - self.phys_base)
            .try_to_usize()
            .expect("mapped offsets fit in usize");
        // SAFETY: the caller of `new` guarantees the mapping covers the
        // range; pointer arithmetic inherits `virt_base`'s provenance.
        unsafe { self.virt_base.add(offset) }
    }

    fn virt_to_phys(&self, ptr: NonNull<u8>) -> A {
        assert!(self.initialized, "the physical mapping is not initialized");
        let offset = ptr.as_ptr().addr() - self.virt_base.as_ptr().addr();
        self.phys_base + A::from_usize(offset)
    }
}

/// Pairs a borrowed buddy allocator with a physical mapping, implementing
/// [`PageAlloc`].
///
/// Mirrors a kernel zone: `Buddy::alloc_pages` plays the role of
/// `alloc_pages_node`, and the mapping stands in for the direct map. Use
/// [`Zone`] instead when the page allocator must live in a `static`.
pub struct BuddyPages<'a, 'b, A: PageFrame, const NR_PAGE_ORDERS: usize, M: PhysMap<Addr = A>> {
    buddy: &'a mut Buddy<'b, A, NR_PAGE_ORDERS>,
    map: M,
}

impl<'a, 'b, A: PageFrame, const NR_PAGE_ORDERS: usize, M: PhysMap<Addr = A>>
    BuddyPages<'a, 'b, A, NR_PAGE_ORDERS, M>
{
    /// Creates a page allocator over `buddy` and `map`.
    pub fn new(buddy: &'a mut Buddy<'b, A, NR_PAGE_ORDERS>, map: M) -> Self {
        Self { buddy, map }
    }

    /// Returns the physical mapping.
    pub fn map(&self) -> &M {
        &self.map
    }
}

impl<A: PageFrame, const NR_PAGE_ORDERS: usize, M: PhysMap<Addr = A>> PhysMap
    for BuddyPages<'_, '_, A, NR_PAGE_ORDERS, M>
{
    type Addr = A;

    fn phys_to_virt(&self, addr: A) -> NonNull<u8> {
        self.map.phys_to_virt(addr)
    }

    fn virt_to_phys(&self, ptr: NonNull<u8>) -> A {
        self.map.virt_to_phys(ptr)
    }
}

impl<A: PageFrame, const NR_PAGE_ORDERS: usize, M: PhysMap<Addr = A>> PageAlloc
    for BuddyPages<'_, '_, A, NR_PAGE_ORDERS, M>
{
    fn page_size(&self) -> usize {
        self.buddy.page_size().try_to_usize().unwrap_or(0)
    }

    fn max_order(&self) -> u8 {
        // `NR_PAGE_ORDERS - 1` via the buddy's canonical constant
        // (mirrors the kernel's `MAX_PAGE_ORDER`).
        <Buddy<'_, A, NR_PAGE_ORDERS>>::MAX_ORDER as u8
    }

    fn alloc_pages(&mut self, order: u8) -> Result<A, Error> {
        self.buddy.alloc_pages(order).map_err(map_buddy_error)
    }

    fn free_pages(&mut self, addr: A, order: u8) -> Result<(), Error> {
        self.buddy.free_pages(addr, order).map_err(map_buddy_error)
    }
}

/// A self-contained page allocator: a buddy arena plus its direct mapping,
/// both owned by value so the whole zone can live in a `static`.
///
/// [`BuddyPages`] borrows the buddy allocator, which a `static` cannot
/// hold; caches and the kernel heap are usually statics themselves and
/// only borrow the page allocator for the duration of a call, so the page
/// allocator is the piece that must be owned and long-lived. `Zone` is
/// that piece:
///
/// ```ignore
/// static ZONE: Mutex<Zone<usize, 11>> = Mutex::new(Zone::uninit());
///
/// // After memory discovery, once the vmemmap is mapped:
/// let mut zone = ZONE.lock().unwrap();
/// // SAFETY: the descriptors and the mapping are valid for the rest of
/// // the system, and all access goes through the lock.
/// unsafe { zone.init(vmemmap, nr_pages, base, page_size, virt_base) }?;
/// zone.buddy_mut().free_memblock(&memblock)?;
///
/// // ... and later, from any cache:
/// let object = CACHE.lock().unwrap().alloc(&mut *zone)?;
/// ```
///
/// Mirrors one zone's page frame management (`struct zone` without
/// watermarks): the buddy allocator plays the role of the free areas and
/// the direct map stands in for `page_address`.
///
/// When the zone and a cache are both behind locks, acquire the cache lock
/// first and then the zone lock, mirroring SLUB's slab-then-zone order,
/// and keep that order at every call site.
pub struct Zone<A: PageFrame, const NR_PAGE_ORDERS: usize> {
    buddy: Buddy<'static, A, NR_PAGE_ORDERS>,
    map: DirectMap<A>,
}

// SAFETY: `Zone` owns the arena and the mapping and never exposes them
// outside its methods. Every method's contract requires externally
// synchronized access (see `Buddy::from_raw_parts`), so moving the zone
// between threads — for example into a lock — cannot create a data race.
unsafe impl<A: PageFrame + Send, const NR_PAGE_ORDERS: usize> Send for Zone<A, NR_PAGE_ORDERS> {}

impl<A: PageFrame, const NR_PAGE_ORDERS: usize> Zone<A, NR_PAGE_ORDERS> {
    /// A `const` placeholder for static definitions.
    pub const fn uninit() -> Self {
        Self {
            buddy: Buddy::uninit(),
            map: DirectMap::uninit(),
        }
    }

    /// Initializes the arena and the mapping in one step.
    ///
    /// Mirrors `free_area_init` plus the direct map setup. The managed
    /// range starts at `base`; memory is not usable until it is handed to
    /// [`Zone::buddy_mut`] with `free_range` or `free_memblock`.
    ///
    /// # Safety
    ///
    /// - `descriptors` and `nr_pages` must satisfy [`Buddy::init`]'s
    ///   contract (valid, aligned, exclusively accessible descriptors that
    ///   outlive the zone);
    /// - `virt_base` must satisfy [`DirectMap::new`]'s contract for the
    ///   range starting at `base`;
    /// - all access to the zone must be externally synchronized for as
    ///   long as it is used.
    ///
    /// # Errors
    ///
    /// Same as [`Buddy::init`].
    pub unsafe fn init(
        &mut self,
        descriptors: NonNull<Page>,
        nr_pages: usize,
        base: A,
        page_size: A,
        virt_base: NonNull<u8>,
    ) -> Result<(), Error> {
        // SAFETY: the caller guarantees the descriptor contract.
        unsafe { self.buddy.init(descriptors, nr_pages, base, page_size) }.map_err(map_buddy_error)?;
        // SAFETY: the caller guarantees the mapping contract, and `base`
        // is the arena start, so the mapping covers every block the buddy
        // hands out.
        unsafe { self.map.init(base, virt_base) };
        Ok(())
    }

    /// Creates an initialized zone from raw parts.
    ///
    /// # Safety
    ///
    /// Same contract as [`Zone::init`].
    ///
    /// # Errors
    ///
    /// Same as [`Buddy::init`].
    pub unsafe fn from_raw_parts(
        descriptors: NonNull<Page>,
        nr_pages: usize,
        base: A,
        page_size: A,
        virt_base: NonNull<u8>,
    ) -> Result<Self, Error> {
        let mut zone = Self::uninit();
        // SAFETY: the caller guarantees both contracts.
        unsafe { zone.init(descriptors, nr_pages, base, page_size, virt_base) }?;
        Ok(zone)
    }

    /// Returns `true` once [`Zone::init`] has run.
    pub const fn is_initialized(&self) -> bool {
        self.buddy.is_initialized()
    }

    /// Returns the underlying buddy allocator.
    pub fn buddy(&self) -> &Buddy<'static, A, NR_PAGE_ORDERS> {
        &self.buddy
    }

    /// Returns the underlying buddy allocator, for example to feed it the
    /// memory left over by the boot allocator.
    pub fn buddy_mut(&mut self) -> &mut Buddy<'static, A, NR_PAGE_ORDERS> {
        &mut self.buddy
    }

    /// Returns the physical mapping.
    pub fn map(&self) -> &DirectMap<A> {
        &self.map
    }
}

impl<A: PageFrame, const NR_PAGE_ORDERS: usize> PhysMap for Zone<A, NR_PAGE_ORDERS> {
    type Addr = A;

    fn phys_to_virt(&self, addr: A) -> NonNull<u8> {
        self.map.phys_to_virt(addr)
    }

    fn virt_to_phys(&self, ptr: NonNull<u8>) -> A {
        self.map.virt_to_phys(ptr)
    }
}

impl<A: PageFrame, const NR_PAGE_ORDERS: usize> PageAlloc for Zone<A, NR_PAGE_ORDERS> {
    fn page_size(&self) -> usize {
        self.buddy.page_size().try_to_usize().unwrap_or(0)
    }

    fn max_order(&self) -> u8 {
        // `NR_PAGE_ORDERS - 1` via the buddy's canonical constant
        // (mirrors the kernel's `MAX_PAGE_ORDER`).
        <Buddy<'static, A, NR_PAGE_ORDERS>>::MAX_ORDER as u8
    }

    fn alloc_pages(&mut self, order: u8) -> Result<A, Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }
        self.buddy.alloc_pages(order).map_err(map_buddy_error)
    }

    fn free_pages(&mut self, addr: A, order: u8) -> Result<(), Error> {
        if !self.is_initialized() {
            return Err(Error::Uninitialized);
        }
        self.buddy.free_pages(addr, order).map_err(map_buddy_error)
    }
}

/// Projects buddy errors onto slab errors.
fn map_buddy_error(error: BuddyError) -> Error {
    match error {
        BuddyError::OutOfMemory => Error::OutOfMemory,
        _ => Error::PageAlloc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_map_round_trips() {
        let mut memory = [0u8; 64];
        let base = memory.as_mut_ptr().addr();
        let virt = NonNull::new(memory.as_mut_ptr()).unwrap();

        // SAFETY: the mapping is the identity over this array.
        let map = unsafe { DirectMap::<usize>::new(base, virt) };

        let ptr = map.phys_to_virt(base + 7);
        assert_eq!(ptr.as_ptr().addr(), base + 7);
        assert_eq!(map.virt_to_phys(ptr), base + 7);

        // The pointer keeps provenance: writing through it works.
        unsafe { ptr.write(42) };
        assert_eq!(memory[7], 42);
    }

    #[test]
    #[should_panic(expected = "address below the mapped base")]
    fn direct_map_rejects_addresses_below_the_base() {
        let mut memory = [0u8; 16];
        let base = memory.as_mut_ptr().addr();
        let virt = NonNull::new(memory.as_mut_ptr()).unwrap();

        // SAFETY: the mapping is the identity over this array.
        let map = unsafe { DirectMap::<usize>::new(base, virt) };
        let _ = map.phys_to_virt(base - 1);
    }

    #[test]
    fn direct_map_init_fills_the_placeholder() {
        let mut memory = [0u8; 64];
        let base = memory.as_mut_ptr().addr();
        let virt = NonNull::new(memory.as_mut_ptr()).unwrap();

        let mut map = DirectMap::<usize>::uninit();
        assert!(!map.is_initialized());

        // SAFETY: the mapping is the identity over this array.
        unsafe { map.init(base, virt) };
        assert!(map.is_initialized());
        assert_eq!(map.phys_base(), base);
        assert_eq!(map.phys_to_virt(base + 7).as_ptr().addr(), base + 7);
    }

    #[test]
    #[should_panic(expected = "the physical mapping is not initialized")]
    fn direct_map_rejects_use_before_init() {
        let map = DirectMap::<usize>::uninit();
        let _ = map.phys_to_virt(0);
    }

    #[test]
    fn zone_uninit_is_inert_and_const() {
        const PLACEHOLDER: Zone<usize, 4> = Zone::uninit();
        assert!(!PLACEHOLDER.is_initialized());

        let mut zone = Zone::<usize, 4>::uninit();
        assert!(matches!(zone.alloc_pages(0), Err(Error::Uninitialized)));
        assert!(matches!(zone.free_pages(0, 0), Err(Error::Uninitialized)));
    }
}
