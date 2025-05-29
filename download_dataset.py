from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="neuralzome/crate",       
    repo_type="dataset",              
    local_dir="neuralzome_crate_local"
)
