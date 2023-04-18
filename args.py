import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for Alpaca-LoRA as a chatbot service"
    )
    # Dataset related.
    parser.add_argument(
        "--base_url",
        help="Hugging Face Hub URL",
        # default="decapoda-research/llama-13b-hf",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )
    parser.add_argument(
        "--ft_ckpt_url",
        help="Hugging Face Hub URL",
        default="tloen/alpaca-lora-7b",
        #default="chansung/gpt4-alpaca-lora-13b-1024-tf-4-17",
        # default="chansung/alpaca-lora-13b",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="PORT number where the app is served",
        default=6006,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of requests to handle at the same time",
        default=1,
        type=int
    )        
    parser.add_argument(
        "--api_open",
        help="Open as API",
        action='store_true'
    )
    parser.add_argument(
        "--share",
        help="Create and share temporary endpoint (useful in Colab env)",
        action='store_true'
    )
    parser.add_argument(
        "--gen_config_path",
        help="path to GenerationConfig file used in batch mode",
        default="configs/gen_config_default.yaml",
        type=str
    )
    parser.add_argument(
        "--gen_config_summarization_path",
        help="path to GenerationConfig file used in context summarization",
        default="configs/gen_config_summarization.yaml",
        type=str
    )
    parser.add_argument(
        "--get_constraints_config_path",
        help="path to ConstraintsConfig file used to constraint user inputs",
        default="configs/constraints_config.yaml",
        type=str
    )
    parser.add_argument(
        "--multi_gpu",
        help="Enable multi gpu mode. This will force not to use Int8 but float16, so you need to check if your system has enough GPU memory",
        action='store_true'
    )
    parser.add_argument(
        "--force_download_ckpt",
        help="Force to download ckpt instead of using cached one",
        action="store_true"
    )
    
    return parser.parse_args()