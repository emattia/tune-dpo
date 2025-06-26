import os
from metaflow import (
    FlowSpec,
    step,
    current,
    Parameter,
    card,
    gpu_profile,
    model,
    environment,
    IncludeFile,
    huggingface_hub,
    checkpoint,
    kubernetes,
    secrets,
    torchrun
)

# from launcher import TorchTune
from metaflow.plugins.torchtune import TorchTune

N_GPU = 8
N_NODES = 4

if N_GPU==8: # tested on H100 80GB
    k8s_config = dict(
        compute_pool="cks-h100",
        cpu=100,
        memory=900 * 1000,
        gpu=N_GPU,
        shared_memory=200 * 1000,
        # This thing needs a security context of `V1Container` with privilage=true to use Infiniband.
        disk=1000 * 1000,
        use_tmpfs=True,
        security_context=dict(
            privileged=True,
        ),
    )

elif N_GPU==4: # tested on a10g, AWS g5.12xlarge
    k8s_config = dict(
        compute_pool="a10g4x",
        cpu=42,
        memory=164 * 1000,
        gpu=N_GPU,
        shared_memory=128 * 1000,
        # This thing needs a security context of `V1Container` with privilage=true to use Infiniband.
        disk=400 * 1000,
        use_tmpfs=True,
    )

else:
    raise ValueError('N_GPU must equal 8 or 4.')


def model_cache_environment(func):
    deco_list = [
        # pypi(
        #     python="3.11.10",
        #     packages={
        #         "huggingface-hub[hf_transfer]": "0.25.2",
        #         "omegaconf": "2.4.0.dev3",
        #     },  # Installing Hugging Face Hub with transfer feature
        # ),
        huggingface_hub(temp_dir_root="metaflow-chkpt/hf_hub"),
        environment(
            vars={
                "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable Hugging Face transfer acceleration
            }
        ),
        secrets(sources=["outerbounds.eddie-hf"]) # TODO: fill your secret
    ]
    for deco in deco_list:
        func = deco(func)
    return func

def training_environment(func):
    deco_list = [
        card(),
        gpu_profile(interval=10),
        # pypi(
        #     python="3.11.10",
        #     packages={
        #         # "wandb": "0.19.5",
        #         "kagglehub": "0.3.6",
        #         "datasets": "3.2.0",
        #         "transformers": "4.48.3",
        #         "torchtune": "0.6.0",
        #         "torch": "2.5.1",
        #         "torchvision": "0.20.1",
        #         "torchao": "0.8.0",
        #         "setuptools": ""
        #     },
        # ),
        environment(
            vars={
                "WANDB_PROJECT": "dpo",
                "WANDB_LOG_MODEL": "false",

                "OMP_NUM_THREADS": "8",
                "TORCH_DIST_INIT_BARRIER": "1",
                # Multinode NCCL settings for Coreweave
                # https://docs.coreweave.com/docs/products/networking/how-to/use-rdma#kubernetes-example
                "NCCL_SOCKET_IFNAME": "eth0", # The network interface name to use for NCCL communication. 
                                              # This should be set to the InfiniBand interface name.

                # Coreweave InfiniBand settings
                "NCCL_IB_HCA": "ibp",         # The InfiniBand host channel adapter (HCA) to use for NCCL communication.
                "UCX_NET_DEVICES": "ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1",        
                                              # The network devices to use for UCX communication. 
                                              # This should be set to the InfiniBand interface name.

                # Nebius InfiniBand settings
                # "NCCL_IB_HCA": "mlx5",
                # "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",

                # "NCCL_DEBUG": "INFO",         # For debugging; remove in production
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func

def inference_environment(func):
    deco_list = [
        card(),
        gpu_profile(interval=10),
        # pypi(
        #     python="3.11.10",
        #     packages={
        #         "kagglehub": "0.3.6",
        #         "datasets": "3.2.0",
        #         "vllm": "0.7.2",
        #         "transformers": "4.48.3",
        #         "torchtune": "0.6.0",
        #         "torch": "2.5.1",
        #         "torchvision": "0.20.1",
        #         "torchao": "0.8.0",
        #         "setuptools": ""
        #     },
        # ),
        environment(
            vars={
                "WANDB_PROJECT": "dpo",
                "WANDB_LOG_MODEL": "false",
                # "NCCL_IB_HCA": "mlx5",
                # "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
                # "SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING": "1",
                # "NCCL_COLLNET_ENABLE": "0",
                "OMP_NUM_THREADS": "8",
                "TORCH_DIST_INIT_BARRIER": "1"
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


class DPOPostTrainMultinodeTPDemo(FlowSpec):

    training_config = IncludeFile(
        "config",
        default="dpo_config_tp.yaml", # TODO: change to desired config.yaml file.
        is_text=True,
    )
    prev_model_key = Parameter(
        "pre-model-key", 
        default=None,
        type=str
    )
    recipe = Parameter(
        "recipe",
        default="dpo_recipe_tp.py", # TODO: change to desired torchtune recipe.
        help="The name of the recipe or .py file that defines the recipe. Metaflow will automatically package .py files in the flow directory."
    )

    # train_split = Parameter(
    #     "train-split",
    #     default="train[:90%]",
    #     help="See https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path"
    # )
    # test_split = Parameter(
    #     "test",
    #     default="train[:90%]",
    #     help="See https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path"
    # )
    
    @step
    def start(self):
        self.next(self.pull_model)

    @model_cache_environment
    @kubernetes(**k8s_config, image='docker.io/eddieob/hf-model-cache')
    @step
    def pull_model(self):
        '''
        Cache the model weights in Metaflow datastore.
        Notice connection to downstream @model use.
        '''
        import yaml
        from omegaconf import OmegaConf

        config = OmegaConf.create(yaml.safe_load(self.training_config))
        config = OmegaConf.to_container(config, resolve=True)
        self.model_name = config["huggingface"]["repo_id"]
        current.run.add_tag("model:%s" % self.model_name)

        if self.prev_model_key is None:
            self.base_model = current.huggingface_hub.snapshot_download(
                repo_id=self.model_name,
                allow_patterns=config["huggingface"]["allow_patterns"],
                max_workers=100,
                repo_type="model",
            )
        else: 
            self.base_model = self.prev_model_key
        self.next(self.train, num_parallel=N_NODES)

    # TODO: Read checkpointing docs at
    # https://docs.metaflow.org/scaling/checkpoint/checkpoint-ml-libraries#checkpointing-pytorch
    # Checkpoints store intermediate state.
    @checkpoint(
        # load_policy="eager", 
        temp_dir_root="metaflow-chkpt-train/loaded_checkpoints"
    )
    @model(
        load=[("base_model", "metaflow-chkpt-train/model")], 
        temp_dir_root="metaflow-chkpt-train/loaded_models"
    )
    @training_environment
    # @kubernetes(**k8s_config, image="docker.io/eddieob/torchtune-train")
    @kubernetes(**k8s_config, image="docker.io/eddieob/coreweave-torchtune-train:mine")
    @torchrun
    @step
    def train(self):
        import yaml

        config = yaml.safe_load(self.training_config)
        config["base_dir"] = current.model.loaded["base_model"]
        if current.checkpoint.is_loaded:
            # If we have a checkpoint loaded because of some failure then 
            # we will also load the recipe checkpoint if it exists. 
            config["base_dir"] = current.checkpoint.directory
            if "recipe_checkpoint_key" in current.checkpoint.info.metadata:
                config["recipe_checkpoint_key"] = current.checkpoint.info.metadata["recipe_checkpoint_key"]
                recipe_checkpoint_path = current.model.load(
                    config["recipe_checkpoint_key"]
                )
                config["checkpointer"]["recipe_checkpoint"] = os.path.join(recipe_checkpoint_path, "recipe_state.pt")
                config["resume_from_checkpoint"] = True
                print("Resuming from checkpoint recipe of task:", current.checkpoint.info.pathspec, recipe_checkpoint_path)            
        config["run_name"] = current.pathspec
        config["output_dir"] = os.path.join(current.tempdir, "output")
        # config["dataset"]["split"] = self.train_split

        tune = TorchTune(
            use_multi_node_config=True, 
            config_overrides={"nproc_per_node": N_GPU}
        )
        tune.run(
            self.recipe,
            config_dict=config,
            # additional_cli_options removed since @torchrun handles this
        )

        # if current.parallel.node_index == 0:
        #     self.dpo_model_ref = current.model.save(
        #         os.path.join(
        #             config["output_dir"],
        #             "epoch_" + str(config["epochs"] - 1),
        #         ),
        #         storage_format="files",
        #     )
        self.next(self.join)

    @step
    def join(self, inputs):
        # self.dpo_model_ref = inputs[0].dpo_model_ref
    #     self.next(self.eval)

    # @card
    # @model(
    #     load=[("dpo_model_ref", "metaflow-chkpt-infer/dpo_model")], 
    #     temp_dir_root="metaflow-chkpt-infer/loaded_models"
    # )
    # @inference_environment
    # @kubernetes(**k8s_config, image="docker.io/eddieob/torchtune-vllm-inference")
    # @step
    # def eval(self):
    #     from dpo_eval import run_eval

    #     self.results = run_eval(
    #         checkpoint_path=current.model.loaded["dpo_model_ref"],
    #         # data_split=self.test_split,
    #         output_dir="results",
    #         max_batches=10,
    #         world_size=N_GPU,
    #         seed=42
    #     )
        self.next(self.end)

    @step
    def end(self):
        """End of flow"""
        # print("Final Model Key:", self.dpo_model_ref)
        pass


if __name__ == "__main__":
    DPOPostTrainMultinodeTPDemo()
