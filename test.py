from omegaconf import OmegaConf

args = OmegaConf.from_cli()
default_cfg = OmegaConf.load(args.config)
model_cfg = OmegaConf.load(args.config1)
#print(cfg)
cfg = OmegaConf.merge(default_cfg,model_cfg)
cfg = OmegaConf.merge(cfg,args)
breakpoint()