yaml_file=resnet18_cifar100  # use a config for cifar100 if you have it; else use your closest
subroot=cls
data_path=$1

# Create config folder if not exist
if [ ! -d "config/mm${subroot}/${yaml_file}" ]; then
    mkdir -p config/mm${subroot}/${yaml_file}
fi

# Download config and checkpoint from mmcls (or place manually if you have them)
mim download mmcls --config ${yaml_file} --dest config/mm${subroot}/${yaml_file}/

# Find checkpoint and config file paths
ckpt_file=$(ls config/mm${subroot}/${yaml_file}/*.pth)
config_file=$(ls config/mm${subroot}/${yaml_file}/*.py)

# Update checkpoint path in config file
sed -i "s#load_from = None#load_from = '${ckpt_file}'#g" $config_file

# Update data path in config file to your CIFAR-100 data folder
sed -i "s#data_prefix = '.*'#data_prefix = '${data_path}/data'#g" $config_file

echo "Config and checkpoint updated:"
echo "Config file: $config_file"
echo "Checkpoint file: $ckpt_file"