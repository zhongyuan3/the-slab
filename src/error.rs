//! The error type returned by fallible slab operations.

/// Errors returned by the slab allocator.
///
/// The variants mirror the failure modes SLUB reports through
/// `slab_err`/`object_err` (`mm/slub.c`) and the allocation failure of
/// `slab_alloc_node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The page allocator has no free block of the requested order.
    OutOfMemory,
    /// An operation that needs a layout was attempted before
    /// [`ObjectCache::init`].
    ///
    /// [`ObjectCache::init`]: crate::cache::ObjectCache::init
    Uninitialized,
    /// [`ObjectCache::init`] was called on a cache that is already
    /// initialized.
    ///
    /// [`ObjectCache::init`]: crate::cache::ObjectCache::init
    AlreadyInitialized,
    /// The object size is zero or smaller than a free list pointer.
    InvalidObjectSize,
    /// The alignment is zero, not a power of two, larger than the page
    /// size, or not satisfiable by any available slab class.
    InvalidAlign,
    /// The page size is zero or not a power of two.
    InvalidPageSize,
    /// No buddy order up to `max_order` can hold even one object.
    ObjectTooLarge,
    /// The pointer does not denote an object of this cache: it is
    /// misaligned, outside the object area, or points into memory that is
    /// not a slab header.
    InvalidPointer,
    /// The pointer belongs to a slab of a different cache.
    CrossCacheFree,
    /// The object is already free.
    DoubleFree,
    /// Slab metadata is corrupted (bad magic/owner/order, broken free list
    /// or partial list).
    CorruptSlab,
    /// Bookkeeping does not match the slabs that can be walked.
    CountMismatch,
    /// The underlying page allocator failed for a reason other than
    /// running out of memory.
    PageAlloc,
    /// An invariant was violated; indicates a bug in the implementation.
    InternalError,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Error::OutOfMemory => "Out of memory",
            Error::Uninitialized => "Object cache is not initialized",
            Error::AlreadyInitialized => "Object cache is already initialized",
            Error::InvalidObjectSize => "Invalid object size",
            Error::InvalidAlign => "Invalid alignment",
            Error::InvalidPageSize => "Invalid page size",
            Error::ObjectTooLarge => "Object does not fit a slab block",
            Error::InvalidPointer => "Invalid object pointer",
            Error::CrossCacheFree => "Object belongs to another cache",
            Error::DoubleFree => "Object is already free",
            Error::CorruptSlab => "Corrupted slab metadata",
            Error::CountMismatch => "Slab accounting mismatch",
            Error::PageAlloc => "Page allocator failure",
            Error::InternalError => "Internal error",
        };
        write!(f, "{s}")
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::string::ToString;

    use super::*;

    #[test]
    fn error_is_copy_and_eq() {
        let error = Error::OutOfMemory;
        let copy = error;
        assert_eq!(error, copy);
        assert_ne!(Error::OutOfMemory, Error::DoubleFree);
    }

    #[test]
    fn display_covers_all_variants() {
        assert_eq!(Error::OutOfMemory.to_string().as_str(), "Out of memory");
        assert_eq!(
            Error::Uninitialized.to_string().as_str(),
            "Object cache is not initialized"
        );
        assert_eq!(
            Error::AlreadyInitialized.to_string().as_str(),
            "Object cache is already initialized"
        );
        assert_eq!(
            Error::InvalidObjectSize.to_string().as_str(),
            "Invalid object size"
        );
        assert_eq!(
            Error::InvalidAlign.to_string().as_str(),
            "Invalid alignment"
        );
        assert_eq!(
            Error::InvalidPageSize.to_string().as_str(),
            "Invalid page size"
        );
        assert_eq!(
            Error::ObjectTooLarge.to_string().as_str(),
            "Object does not fit a slab block"
        );
        assert_eq!(
            Error::InvalidPointer.to_string().as_str(),
            "Invalid object pointer"
        );
        assert_eq!(
            Error::CrossCacheFree.to_string().as_str(),
            "Object belongs to another cache"
        );
        assert_eq!(
            Error::DoubleFree.to_string().as_str(),
            "Object is already free"
        );
        assert_eq!(
            Error::CorruptSlab.to_string().as_str(),
            "Corrupted slab metadata"
        );
        assert_eq!(
            Error::CountMismatch.to_string().as_str(),
            "Slab accounting mismatch"
        );
        assert_eq!(
            Error::PageAlloc.to_string().as_str(),
            "Page allocator failure"
        );
        assert_eq!(Error::InternalError.to_string().as_str(), "Internal error");
    }
}
