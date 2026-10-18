//! Adapters from [`the_buddy_system::Buddy`] to the [`PageAlloc`] trait.
//!
//! This module is only compiled with the `buddy` feature. The slab core in
//! [`crate::pages`] knows nothing about the buddy allocator, so these
//! adapters are the single integration point:
//!
//! - [`BuddyPages`] borrows a buddy allocator and a [`DirectMap`], which
//!   suits one-shot or scoped use;
//! - [`Zone`] owns both by value and can therefore live in a `static`,
//!   which is how a kernel keeps its page allocator.
//!
//! The boot handoff is explicit and stays outside this module:
//! `zone.free_ranges(mb.free_mem_ranges(flags))` passes a boot allocator's
//! free ranges straight to the owned arena. Neither the slab nor the buddy
//! crate depends on a boot allocator.

use core::ptr::NonNull;

use the_buddy_system::Buddy;
use the_buddy_system::Error as BuddyError;
use the_buddy_system::Page;
use the_mm_core::PageFrame;

use crate::error::Error;
use crate::pages::DirectMap;
use crate::pages::PageAlloc;
use crate::pages::PhysMap;

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

    fn try_phys_to_virt(&self, addr: A) -> Option<NonNull<u8>> {
        self.map.try_phys_to_virt(addr)
    }

    fn try_virt_to_phys(&self, ptr: NonNull<u8>) -> Option<A> {
        self.map.try_virt_to_phys(ptr)
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
/// zone.free_ranges(memblock.free_mem_ranges(flags))?;
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
    /// the zone with [`Zone::free_range`] or [`Zone::free_ranges`].
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
        unsafe { self.buddy.init(descriptors, nr_pages, base, page_size) }
            .map_err(map_buddy_error)?;
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

    /// Returns the underlying buddy allocator, for example to inspect its
    /// free lists or run [`Buddy::validate`].
    pub fn buddy_mut(&mut self) -> &mut Buddy<'static, A, NR_PAGE_ORDERS> {
        &mut self.buddy
    }

    /// Returns the physical mapping.
    pub fn map(&self) -> &DirectMap<A> {
        &self.map
    }

    /// Frees `[start, end)` into the underlying buddy allocator.
    ///
    /// A pass-through to [`Buddy::free_range`] so that callers do not need
    /// to reach through [`Zone::buddy_mut`] for the common boot step.
    pub fn free_range(&mut self, start: A, end: A) -> Result<(), Error> {
        self.buddy.free_range(start, end).map_err(map_buddy_error)
    }

    /// Frees each `[start, end)` range into the underlying buddy allocator.
    ///
    /// A pass-through to [`Buddy::free_ranges`], useful for a boot
    /// allocator's free ranges or a range snapshot. The caller chooses the
    /// source, so the slab crate stays independent of any boot allocator.
    pub fn free_ranges<I>(&mut self, ranges: I) -> Result<(), Error>
    where
        I: IntoIterator<Item = (A, A)>,
    {
        self.buddy.free_ranges(ranges).map_err(map_buddy_error)
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

    fn try_phys_to_virt(&self, addr: A) -> Option<NonNull<u8>> {
        self.map.try_phys_to_virt(addr)
    }

    fn try_virt_to_phys(&self, ptr: NonNull<u8>) -> Option<A> {
        self.map.try_virt_to_phys(ptr)
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
    fn zone_uninit_is_inert_and_const() {
        const PLACEHOLDER: Zone<usize, 4> = Zone::uninit();
        assert!(!PLACEHOLDER.is_initialized());

        let mut zone = Zone::<usize, 4>::uninit();
        assert!(matches!(zone.alloc_pages(0), Err(Error::Uninitialized)));
        assert!(matches!(zone.free_pages(0, 0), Err(Error::Uninitialized)));
    }
}
