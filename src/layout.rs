//! Object and slab layout arithmetic.
//!
//! Mirrors the layout half of SLUB's `calculate_order` and
//! `calculate_sizes` (`mm/slub.c`): the stride between objects, the offset
//! of the first object (after the slab header), the number of objects per
//! slab, and the buddy order of the slab block.
//!
//! Each slab block starts with a slab header (see `crate::header`)
//! followed by the objects; the free list pointers live inside the free
//! objects themselves, exactly like SLUB's `set_freepointer`.

use core::cmp::max;
use core::mem::align_of;
use core::mem::size_of;

use crate::error::Error;

/// Alignment required for the free list pointer stored in free objects.
const FREE_POINTER_ALIGN: usize = align_of::<*mut u8>();

/// A computed slab layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabLayout {
    pub(crate) object_size: usize,
    pub(crate) align: usize,
    pub(crate) stride: usize,
    pub(crate) first_offset: usize,
    pub(crate) objects: usize,
    pub(crate) order: u8,
    pub(crate) slab_bytes: usize,
}

impl SlabLayout {
    /// The layout of an uninitialized cache.
    pub(crate) const EMPTY: SlabLayout = SlabLayout {
        object_size: 0,
        align: 0,
        stride: 0,
        first_offset: 0,
        objects: 0,
        order: 0,
        slab_bytes: 0,
    };

    /// Computes the layout for objects of `object_size` bytes with `align`
    /// alignment, carved out of buddy blocks of `page_size << order`.
    ///
    /// `header_size` bytes at the start of every block are reserved for
    /// the slab header. The smallest order whose slab holds at least `min_objects` objects wins; when no order
    /// reaches that, the order holding the most objects is used, mirroring
    /// SLUB's relaxation for large objects. Objects are laid out on a
    /// `stride` grid so every object is aligned.
    pub(crate) fn new(
        header_size: usize,
        object_size: usize,
        align: usize,
        page_size: usize,
        max_order: u8,
        min_objects: usize,
    ) -> Result<Self, Error> {
        if object_size < size_of::<*mut u8>() {
            return Err(Error::InvalidObjectSize);
        }
        if align == 0 || !align.is_power_of_two() {
            return Err(Error::InvalidAlign);
        }
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(Error::InvalidPageSize);
        }
        if align > page_size {
            return Err(Error::InvalidAlign);
        }
        if max_order as usize >= usize::BITS as usize {
            return Err(Error::ObjectTooLarge);
        }

        // The free list pointer inside an object forces pointer alignment,
        // exactly like SLUB's `calculate_alignment` raising `s->align`.
        let align = max(align, FREE_POINTER_ALIGN);
        let stride = align_up(object_size, align).ok_or(Error::ObjectTooLarge)?;
        // TODO(color): rotate this offset per slab for cache coloring, as
        // SLUB's `s->colour` does, keeping it aligned to `align`.
        let first_offset = align_up(header_size, align).ok_or(Error::InvalidObjectSize)?;

        let mut fallback: Option<(u8, usize, usize)> = None;
        for order in 0..=max_order {
            let Some(slab_bytes) = page_size.checked_shl(order as u32) else {
                break;
            };
            if slab_bytes <= first_offset {
                continue;
            }
            let objects = (slab_bytes - first_offset) / stride;
            if objects == 0 {
                continue;
            }
            if objects >= min_objects {
                return Ok(Self {
                    object_size,
                    align,
                    stride,
                    first_offset,
                    objects,
                    order,
                    slab_bytes,
                });
            }
            fallback = Some((order, objects, slab_bytes));
        }

        let (order, objects, slab_bytes) = fallback.ok_or(Error::ObjectTooLarge)?;
        Ok(Self {
            object_size,
            align,
            stride,
            first_offset,
            objects,
            order,
            slab_bytes,
        })
    }

    /// Returns the requested object size in bytes.
    pub const fn object_size(&self) -> usize {
        self.object_size
    }

    /// Returns the effective object alignment (at least pointer alignment).
    pub const fn align(&self) -> usize {
        self.align
    }

    /// Returns the distance between consecutive objects.
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the offset of the first object in a slab block.
    pub const fn first_offset(&self) -> usize {
        self.first_offset
    }

    /// Returns the number of objects per slab.
    pub const fn objects(&self) -> usize {
        self.objects
    }

    /// Returns the buddy order of a slab block (`2^order` pages).
    pub const fn order(&self) -> u8 {
        self.order
    }

    /// Returns the size of a slab block in bytes.
    pub const fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }
}

/// Rounds `value` up to the next multiple of `align` (a power of two),
/// saturating to `None` on overflow.
fn align_up(value: usize, align: usize) -> Option<usize> {
    let mask = align - 1;
    if value > usize::MAX - mask {
        None
    } else {
        Some((value + mask) & !mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: usize = 56;
    const PAGE: usize = 4096;

    #[test]
    fn small_objects_fit_one_page() {
        let layout = SlabLayout::new(HEADER, 8, 8, PAGE, 10, 4).unwrap();
        assert_eq!(layout.first_offset(), 56);
        assert_eq!(layout.stride(), 8);
        assert_eq!(layout.order(), 0);
        assert_eq!(layout.objects(), (PAGE - 56) / 8);
        assert_eq!(layout.objects(), 505);
    }

    #[test]
    fn unaligned_sizes_are_padded_to_the_alignment() {
        let layout = SlabLayout::new(HEADER, 100, 8, PAGE, 10, 4).unwrap();
        assert_eq!(layout.stride(), 104);
        assert_eq!(layout.objects(), (PAGE - 56) / 104);
    }

    #[test]
    fn small_alignment_is_raised_to_pointer_alignment() {
        let layout = SlabLayout::new(HEADER, 24, 1, PAGE, 10, 4).unwrap();
        assert_eq!(layout.align(), align_of::<*mut u8>());
        assert_eq!(layout.stride(), 24);
    }

    #[test]
    fn large_alignment_is_preserved_in_the_first_object() {
        let layout = SlabLayout::new(HEADER, 64, 256, PAGE, 10, 4).unwrap();
        assert_eq!(layout.align(), 256);
        assert_eq!(layout.first_offset(), 256);
        assert_eq!(layout.stride(), 256);
        assert_eq!(layout.objects(), (PAGE - 256) / 256);
        assert_eq!(layout.objects(), 15);
    }

    #[test]
    fn min_objects_drives_the_order_for_medium_objects() {
        // One page holds a single 2 KiB object; SLUB-style relaxation grows
        // the block until `min_objects` (4) fit.
        let layout = SlabLayout::new(HEADER, 2048, 8, PAGE, 10, 4).unwrap();
        assert_eq!(layout.order(), 2);
        assert_eq!(layout.slab_bytes(), 4 * PAGE);
        assert_eq!(layout.objects(), (4 * PAGE - 56) / 2048);
        assert!(layout.objects() >= 4);
    }

    #[test]
    fn huge_objects_fall_back_to_the_largest_order() {
        // No order reaches `min_objects`; the largest block with the most
        // objects wins instead of failing.
        let layout = SlabLayout::new(HEADER, 4096, 8, PAGE, 3, 4).unwrap();
        assert_eq!(layout.order(), 3);
        assert_eq!(layout.objects(), 7);
    }

    #[test]
    fn object_larger_than_the_largest_block_is_rejected() {
        let error = SlabLayout::new(HEADER, 1 << 20, 8, PAGE, 2, 4).unwrap_err();
        assert_eq!(error, Error::ObjectTooLarge);
    }

    #[test]
    fn page_aligned_objects_do_not_fit_with_a_header() {
        // The object and the header both want the full first page.
        let error = SlabLayout::new(HEADER, PAGE, PAGE, PAGE, 0, 4).unwrap_err();
        assert_eq!(error, Error::ObjectTooLarge);
    }

    #[test]
    fn tiny_objects_are_rejected() {
        assert_eq!(
            SlabLayout::new(HEADER, 4, 8, PAGE, 10, 4).unwrap_err(),
            Error::InvalidObjectSize
        );
        assert_eq!(
            SlabLayout::new(HEADER, 0, 8, PAGE, 10, 4).unwrap_err(),
            Error::InvalidObjectSize
        );
    }

    #[test]
    fn bad_alignment_is_rejected() {
        assert_eq!(
            SlabLayout::new(HEADER, 32, 0, PAGE, 10, 4).unwrap_err(),
            Error::InvalidAlign
        );
        assert_eq!(
            SlabLayout::new(HEADER, 32, 3, PAGE, 10, 4).unwrap_err(),
            Error::InvalidAlign
        );
        assert_eq!(
            SlabLayout::new(HEADER, 32, 8192, PAGE, 10, 4).unwrap_err(),
            Error::InvalidAlign
        );
    }

    #[test]
    fn bad_page_size_is_rejected() {
        assert_eq!(
            SlabLayout::new(HEADER, 32, 8, 0, 10, 4).unwrap_err(),
            Error::InvalidPageSize
        );
        assert_eq!(
            SlabLayout::new(HEADER, 32, 8, 3000, 10, 4).unwrap_err(),
            Error::InvalidPageSize
        );
    }

    #[test]
    fn every_object_is_within_the_block_and_aligned() {
        for (size, align) in [(8, 8), (24, 8), (100, 8), (512, 64), (2048, 256)] {
            let layout = SlabLayout::new(HEADER, size, align, PAGE, 5, 4).unwrap();
            assert!(layout.first_offset() % layout.align() == 0);
            assert!(layout.stride() % layout.align() == 0);
            let last = layout.first_offset() + (layout.objects() - 1) * layout.stride();
            assert!(last + layout.object_size() <= layout.slab_bytes());
        }
    }
}
