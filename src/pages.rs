//! The page allocator and address mapping abstractions.
//!
//! A slab cache never touches a concrete page allocator directly: it asks a
//! [`PageAlloc`] for naturally aligned blocks of `2^order` pages and maps
//! the returned addresses through a [`PhysMap`]. This mirrors the kernel's
//! split between the slab allocator and the page allocator (`new_slab` →
//! `alloc_pages_node`) and keeps physical-address handling out of the slab
//! core.
//!
//! This module contains no buddy-specific code, so `the-slab` can be used
//! with any page allocator. The adapters that implement [`PageAlloc`] over
//! `the_buddy_system::Buddy` live in the crate's `buddy` module behind the
//! `buddy` feature:
//!
//! - `BuddyPages` borrows the buddy allocator and a mapping; it suits
//!   one-shot or scoped use, such as tests;
//! - `Zone` owns both by value and can therefore live in a `static`, which
//!   is how a kernel keeps its page allocator.

use core::ptr::NonNull;

use the_mm_core::PageFrame;

use crate::Error;

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

    /// Checked variant of [`PhysMap::phys_to_virt`].
    ///
    /// Returns `None` when `addr` is outside the mapped range or the mapping
    /// is not initialized. The default implementation delegates to
    /// [`PhysMap::phys_to_virt`], so implementers whose mapping cannot fail
    /// need not override it; mappings that can reject addresses should.
    fn try_phys_to_virt(&self, addr: Self::Addr) -> Option<NonNull<u8>> {
        Some(self.phys_to_virt(addr))
    }

    /// Checked variant of [`PhysMap::virt_to_phys`].
    ///
    /// Returns `None` when `ptr` is outside the mapped range or the mapping
    /// is not initialized. The default implementation delegates to
    /// [`PhysMap::virt_to_phys`], so implementers whose mapping cannot fail
    /// need not override it; mappings that can reject pointers should.
    fn try_virt_to_phys(&self, ptr: NonNull<u8>) -> Option<Self::Addr> {
        Some(self.virt_to_phys(ptr))
    }
}

/// A page allocator handing out naturally aligned blocks of `2^order`
/// pages.
///
/// Mirrors the page allocator interface SLUB uses (`alloc_pages_node` and
/// `__free_pages`), without `GFP` flags or zone selection. Implement it over
/// any physical page source; the crate's `buddy` module provides adapters
/// for `the-buddy-system` behind the `buddy` feature.
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

    fn try_phys_to_virt(&self, addr: A) -> Option<NonNull<u8>> {
        if !self.initialized || addr < self.phys_base {
            return None;
        }
        let offset = (addr - self.phys_base).try_to_usize()?;
        // SAFETY: the caller of `new` guarantees the mapping covers every
        // address the paired page allocator hands out.
        Some(unsafe { self.virt_base.add(offset) })
    }

    fn try_virt_to_phys(&self, ptr: NonNull<u8>) -> Option<A> {
        if !self.initialized {
            return None;
        }
        let virt_base = self.virt_base.as_ptr().addr();
        let offset = ptr.as_ptr().addr().checked_sub(virt_base)?;
        let phys_offset = A::try_from_usize(offset)?;
        if self.phys_base > A::MAX - phys_offset {
            return None;
        }
        Some(self.phys_base + phys_offset)
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
    fn direct_map_checked_translation_rejects_out_of_range() {
        let mut memory = [0u8; 64];
        let base = memory.as_mut_ptr().addr();
        let virt = NonNull::new(memory.as_mut_ptr()).unwrap();

        // SAFETY: the mapping is the identity over this array.
        let map = unsafe { DirectMap::<usize>::new(base, virt) };

        assert_eq!(
            map.try_phys_to_virt(base + 7).unwrap().as_ptr().addr(),
            base + 7
        );
        assert!(map.try_phys_to_virt(base - 1).is_none());

        let ptr = map.phys_to_virt(base + 7);
        assert_eq!(map.try_virt_to_phys(ptr), Some(base + 7));

        // A pointer below the mapping is rejected instead of underflowing.
        let below = NonNull::new((base - 1) as *mut u8).unwrap();
        assert!(map.try_virt_to_phys(below).is_none());
    }

    #[test]
    fn direct_map_checked_translation_rejects_uninitialized() {
        let map = DirectMap::<usize>::uninit();
        assert!(map.try_phys_to_virt(0).is_none());
        assert!(map.try_virt_to_phys(NonNull::dangling()).is_none());
    }
}
