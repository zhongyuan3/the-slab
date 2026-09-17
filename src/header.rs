//! The per-slab metadata stored at the start of every slab block.

use core::ptr::NonNull;

/// Magic written into every slab header and checked by `free` and
/// `validate`.
pub(crate) const SLAB_MAGIC: u32 = 0x5AB0_9001;

/// The metadata of one slab, overlaid on the first bytes of its block.
///
/// The kernel overlays `struct slab` on `struct page`; without a vmemmap
/// this crate stores the metadata at the start of the block itself,
/// followed by the objects. The header keeps the free list head, the
/// in-use count, the identity of the owning cache and the partial list
/// links.
///
/// `magic` is deliberately the first field: `KmallocCaches::kfree` reads
/// the first word of the page containing a pointer to tell a slab block
/// (this header) from a large allocation tag, so the two tag layouts must
/// start with the same magic field.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct SlabHeader {
    /// Magic checked by `free` and `validate`.
    pub(crate) magic: u32,
    /// Buddy order of the block (cross-checked against the cache layout).
    pub(crate) order: u8,
    /// `true` while the slab is linked into the cache's partial list.
    pub(crate) on_partial: bool,
    /// Identity of the owning cache (its address, erased); lets `free`
    /// reject pointers passed to the wrong cache.
    pub(crate) owner: *const (),
    /// Number of objects currently handed out.
    pub(crate) inuse: usize,
    /// Number of objects in the block.
    pub(crate) objects: usize,
    /// Head of the in-object free list.
    pub(crate) freelist: Option<NonNull<u8>>,
    /// Next slab in the partial list.
    pub(crate) next: Option<NonNull<SlabHeader>>,
    /// Previous slab in the partial list.
    pub(crate) prev: Option<NonNull<SlabHeader>>,
}
