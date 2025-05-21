use anyhow::{Context, Result};
use std::marker::PhantomData;

/// Defines the core `Transform` trait for composable data processing pipelines.
///
/// The `Transform<I, O>` trait represents a stateless operation for
/// converting an input of type `I` to an output of type `O`.
/// Multiple `Transform` steps can be chained together via `.then(...)`
/// to form a single, inlined preprocessing pipeline with zero runtime.
///
/// Note: `then()` works only when:
/// 1. **Types align**: `self: Transform<I, O>`, `next: Transform<O, M>`
/// 2. **Owned**: `Self::Sized` (no trait objects, must be concrete)
/// 3. **Thread-safe**: intermediate and output types must be `Send`
pub trait Transform<I, O>: Send + Sync {
    /// Applies the transformation to the input
    fn apply(&self, input: I) -> Result<O>;

    #[inline]
    fn then<T, M>(self, next: T) -> Chain<Self, T, O>
    where
        Self: Sized,
        T: Transform<O, M>,
        O: Send,
        M: Send,
    {
        Chain {
            first: self,
            second: next,
            _marker: PhantomData,
        }
    }
}

/// A chain of two transforms (`A` -> `B`)
/// - `PhantomData<M>` enforces intermediate type alignment.
#[derive(Debug)]
pub struct Chain<A, B, M> {
    first: A,
    second: B,
    _marker: PhantomData<fn() -> M>,
}

impl<A, B, M> Chain<A, B, M> {
    /// Creates a new transform chain.
    /// Use [`Transform::then`] for better ergonomics. `Chain::new` is
    /// useful when building pipelines dynamically (e.g., configurations
    /// loaded from JSON/YAML).
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            _marker: PhantomData,
        }
    }
}

impl<I, M, O, A, B> Transform<I, O> for Chain<A, B, M>
where
    A: Transform<I, M>,
    B: Transform<M, O>,
    M: Send,
{
    fn apply(&self, input: I) -> Result<O> {
        self.first
            .apply(input)
            .and_then(|mid| self.second.apply(mid))
            .with_context(|| {
                format!(
                    "Transform chain failed: {} → {} → {}",
                    std::any::type_name::<A>(),
                    std::any::type_name::<B>(),
                    std::any::type_name::<O>()
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    struct ToUpper;
    impl Transform<String, String> for ToUpper {
        fn apply(&self, input: String) -> Result<String> {
            Ok(input.to_uppercase())
        }
    }

    struct CountBytes;
    impl Transform<String, usize> for CountBytes {
        fn apply(&self, input: String) -> Result<usize> {
            Ok(input.len())
        }
    }

    #[test]
    fn test_pipeline_construction_using_then() -> Result<()> {
        let pipeline = ToUpper.then(CountBytes);
        assert_eq!(pipeline.apply("hello".to_string())?, 5);
        Ok(())
    }

    #[test]
    fn test_pipeline_construction_using_chain() -> Result<()> {
        let chain = Chain::new(ToUpper, CountBytes);
        assert_eq!(chain.apply("hello".to_string())?, 5); // "HELLO".len()
        Ok(())
    }

    #[test]
    fn test_pipeline_chain_error_context() {
        struct Fail;
        impl Transform<String, String> for Fail {
            fn apply(&self, _: String) -> Result<String> {
                Err(anyhow!("Test error"))
            }
        }

        let chain = Chain::new(ToUpper, Fail);
        let err = chain.apply("test".to_string()).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("Transform chain failed"));
        assert!(msg.contains("ToUpper"));
        assert!(msg.contains("Fail"));
    }
}
