from ttbrats.core.opt import Opts
from ttbrats.segmentation.pipeline import Pipeline

if __name__=="__main__":
    opts = Opts.parse("opt.yaml")
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()