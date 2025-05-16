use anyhow::Result;

pub trait Transform<I, O>: Send + Sync {
    fn apply(&self, input: I) -> Result<O>;
}
