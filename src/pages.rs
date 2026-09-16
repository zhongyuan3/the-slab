//! The page allocator and address mapping abstractions.
//!
//! A slab cache never touches the buddy allocator directly: it asks a
//! [`PageAlloc`] for naturally aligned blocks of `2^order` pages and maps
//! the returned addresses through a [`PhysMap`]. This mirrors the kernel's
//! split between the slab allocator and the page allocator (`new_slab` →
//! `alloc_pages_node`) and keeps physical-address handling out of the slab
//! core.

use core::ptr::NonNull;

use the_buddy_system::Buddy;
use the_buddy_system::Error as BuddyError;
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
#[derive(Clone, Copy, Debug)]
pub struct DirectMap<A: PageFrame> {
    phys_base: A,
    virt_base: NonNull<u8>,
}

impl<A: PageFrame> DirectMap<A> {
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
        }
    }

    /// Returns the physical address the mapping starts at.
    pub fn phys_base(&self) -> A {
        self.phys_base
    }
}

impl<A: PageFrame> PhysMap for DirectMap<A> {
    type Addr = A;

    fn phys_to_virt(&self, addr: A) -> NonNull<u8> {
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
        let offset = ptr.as_ptr().addr() - self.virt_base.as_ptr().addr();
        self.phys_base + A::from_usize(offset)
    }
}

/// Pairs a buddy allocator with a physical mapping, implementing
/// [`PageAlloc`].
///
/// Mirrors a kernel zone: `Buddy::alloc_pages` plays the role of
/// `alloc_pages_node`, and the mapping stands in for the direct map.
pub struct BuddyPages<'a, 'b, A: PageFrame, const MAX_ORDER: usize, M: PhysMap<Addr = A>> {
    buddy: &'a mut Buddy<'b, A, MAX_ORDER>,
    map: M,
}

impl<'a, 'b, A: PageFrame, const MAX_ORDER: usize, M: PhysMap<Addr = A>>
    BuddyPages<'a, 'b, A, MAX_ORDER, M>
{
    /// Creates a page allocator over `buddy` and `map`.
    pub fn new(buddy: &'a mut Buddy<'b, A, MAX_ORDER>, map: M) -> Self {
        Self { buddy, map }
    }

    /// Returns the physical mapping.
    pub fn map(&self) -> &M {
        &self.map
    }
}

impl<A: PageFrame, const MAX_ORDER: usize, M: PhysMap<Addr = A>> PhysMap
    for BuddyPages<'_, '_, A, MAX_ORDER, M>
{
    type Addr = A;

    fn phys_to_virt(&self, addr: A) -> NonNull<u8> {
        self.map.phys_to_virt(addr)
    }

    fn virt_to_phys(&self, ptr: NonNull<u8>) -> A {
        self.map.virt_to_phys(ptr)
    }
}

impl<A: PageFrame, const MAX_ORDER: usize, M: PhysMap<Addr = A>> PageAlloc
    for BuddyPages<'_, '_, A, MAX_ORDER, M>
{
    fn page_size(&self) -> usize {
        self.buddy.page_size().try_to_usize().unwrap_or(0)
    }

    fn max_order(&self) -> u8 {
        (MAX_ORDER - 1) as u8
    }

    fn alloc_pages(&mut self, order: u8) -> Result<A, Error> {
        self.buddy.alloc_pages(order).map_err(map_buddy_error)
    }

    fn free_pages(&mut self, addr: A, order: u8) -> Result<(), Error> {
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
}
