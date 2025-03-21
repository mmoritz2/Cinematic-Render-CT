#!/usr/bin/env python3
"""
Cloud GPU module for the Cinematic CT Viewer
Supports AWS, GCP, and Azure for remote training
"""

import os
import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path

# Import necessary libraries conditionally - these would be installed when needed
try:
    import boto3  # AWS SDK
except ImportError:
    boto3 = None

try:
    from google.cloud import compute_v1  # GCP SDK
except ImportError:
    compute_v1 = None

try: 
    from azure.identity import DefaultAzureCredential  # Azure SDK
    from azure.mgmt.compute import ComputeManagementClient
except ImportError:
    DefaultAzureCredential = None
    ComputeManagementClient = None


def check_cloud_dependencies(provider):
    """
    Check and install necessary dependencies for the specified cloud provider
    """
    if provider == 'aws' and boto3 is None:
        print("Installing AWS SDK (boto3)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
        return True
    
    elif provider == 'gcp' and compute_v1 is None:
        print("Installing Google Cloud SDK...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-compute"])
        return True
    
    elif provider == 'azure' and (DefaultAzureCredential is None or ComputeManagementClient is None):
        print("Installing Azure SDK...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "azure-identity", "azure-mgmt-compute"])
        return True
    
    return False


def prepare_cloud_job(provider, instance_type, region):
    """
    Prepare a cloud GPU instance for training
    
    Args:
        provider: Cloud provider (aws, gcp, azure)
        instance_type: Instance type to use
        region: Region to launch the instance in
        
    Returns:
        instance_id: ID of the created instance
    """
    # Check and install dependencies if needed
    deps_installed = check_cloud_dependencies(provider)
    if deps_installed:
        # Reload modules if we just installed them
        if provider == 'aws':
            import boto3
        elif provider == 'gcp':
            from google.cloud import compute_v1
        elif provider == 'azure':
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
    
    # Prepare startup script that will be run on the instance
    startup_script = _generate_startup_script(provider)
    
    print(f"Setting up {provider} GPU instance ({instance_type}) in {region}...")
    
    instance_id = None
    if provider == 'aws':
        instance_id = _setup_aws_instance(instance_type, region, startup_script)
    elif provider == 'gcp':
        instance_id = _setup_gcp_instance(instance_type, region, startup_script)
    elif provider == 'azure':
        instance_id = _setup_azure_instance(instance_type, region, startup_script)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    
    print(f"Cloud instance {instance_id} is being prepared. This may take a few minutes...")
    
    # Wait for instance to be ready
    _wait_for_instance_ready(provider, instance_id, region)
    
    return instance_id


def upload_data(provider, instance_id, data_dir, iterations):
    """
    Upload data to the cloud instance
    
    Args:
        provider: Cloud provider (aws, gcp, azure)
        instance_id: ID of the created instance
        data_dir: Directory containing the data to upload
        iterations: Number of training iterations
    """
    if provider == 'aws':
        _upload_data_aws(instance_id, data_dir, iterations)
    elif provider == 'gcp':
        _upload_data_gcp(instance_id, data_dir, iterations)
    elif provider == 'azure':
        _upload_data_azure(instance_id, data_dir, iterations)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")


def start_training(provider, instance_id, iterations):
    """
    Start the training job on the cloud instance
    
    Args:
        provider: Cloud provider (aws, gcp, azure)
        instance_id: ID of the created instance
        iterations: Number of training iterations
    """
    if provider == 'aws':
        _start_training_aws(instance_id, iterations)
    elif provider == 'gcp':
        _start_training_gcp(instance_id, iterations)
    elif provider == 'azure':
        _start_training_azure(instance_id, iterations)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")


def download_results(provider, instance_id, model_output_dir, compressed_output_dir):
    """
    Download results from the cloud instance
    
    Args:
        provider: Cloud provider (aws, gcp, azure)
        instance_id: ID of the created instance
        model_output_dir: Directory to save the model output
        compressed_output_dir: Directory to save the compressed output
    """
    if provider == 'aws':
        _download_results_aws(instance_id, model_output_dir, compressed_output_dir)
    elif provider == 'gcp':
        _download_results_gcp(instance_id, model_output_dir, compressed_output_dir)
    elif provider == 'azure':
        _download_results_azure(instance_id, model_output_dir, compressed_output_dir)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    
    # Cleanup cloud resources
    _cleanup_cloud_resources(provider, instance_id)


# ------------------- AWS Implementation -------------------

def _setup_aws_instance(instance_type, region, startup_script):
    """Set up an AWS EC2 instance with GPU"""
    import boto3
    
    # Create EC2 client
    ec2 = boto3.client('ec2', region_name=region)
    
    # Create key pair for SSH access
    key_name = f"cinematic-render-{int(time.time())}"
    key_pair = ec2.create_key_pair(KeyName=key_name)
    
    # Save private key to file
    private_key = key_pair['KeyMaterial']
    key_file = os.path.expanduser(f"~/.ssh/{key_name}.pem")
    with open(key_file, 'w') as f:
        f.write(private_key)
    os.chmod(key_file, 0o600)
    
    print(f"Created SSH key pair {key_name} and saved to {key_file}")
    
    # Create security group
    sg_name = f"cinematic-render-sg-{int(time.time())}"
    sg = ec2.create_security_group(
        GroupName=sg_name,
        Description="Security group for Cinematic Render CT"
    )
    
    # Add SSH and HTTPS inbound rules
    ec2.authorize_security_group_ingress(
        GroupId=sg['GroupId'],
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            },
            {
                'IpProtocol': 'tcp',
                'FromPort': 443,
                'ToPort': 443,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            }
        ]
    )
    
    # Find latest deep learning AMI
    response = ec2.describe_images(
        Owners=['amazon'],
        Filters=[
            {'Name': 'name', 'Values': ['Deep Learning AMI GPU PyTorch*']},
            {'Name': 'state', 'Values': ['available']},
            {'Name': 'architecture', 'Values': ['x86_64']}
        ]
    )
    
    # Sort by creation date to get the latest
    amis = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)
    ami_id = amis[0]['ImageId'] if amis else 'ami-0c7217cdde317cfec'  # Fallback to known AMI
    
    print(f"Using AMI: {ami_id}")
    
    # Launch the instance
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[sg['GroupId']],
        UserData=startup_script,
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 100,  # GB
                    'VolumeType': 'gp2'
                }
            }
        ]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    
    # Tag the instance
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[{'Key': 'Name', 'Value': 'CinematicRenderCT'}]
    )
    
    # Save instance info to local file for later use
    instance_info = {
        'provider': 'aws',
        'instance_id': instance_id,
        'key_name': key_name,
        'key_file': key_file,
        'region': region,
        'security_group_id': sg['GroupId']
    }
    
    with open('cloud_instance_info.json', 'w') as f:
        json.dump(instance_info, f)
    
    return instance_id


def _upload_data_aws(instance_id, data_dir, iterations):
    """Upload data to AWS instance"""
    import boto3
    
    # Load instance info
    with open('cloud_instance_info.json', 'r') as f:
        instance_info = json.load(f)
    
    # Get instance public IP
    ec2 = boto3.client('ec2', region_name=instance_info['region'])
    response = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    # Create remote directories
    key_file = instance_info['key_file']
    ssh_cmd = f"ssh -i {key_file} -o StrictHostKeyChecking=no ubuntu@{public_ip}"
    
    # Create directories
    subprocess.run(f"{ssh_cmd} 'mkdir -p ~/cinematic-render/scene'", shell=True, check=True)
    
    # Upload data using scp
    scp_cmd = f"scp -i {key_file} -r {data_dir}/* ubuntu@{public_ip}:~/cinematic-render/scene/"
    subprocess.run(scp_cmd, shell=True, check=True)
    
    # Create config file with iterations
    config = {
        'iterations': iterations
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump(config, f)
        config_file = f.name
    
    # Upload config
    scp_cmd = f"scp -i {key_file} {config_file} ubuntu@{public_ip}:~/cinematic-render/config.json"
    subprocess.run(scp_cmd, shell=True, check=True)
    
    # Remove temp file
    os.unlink(config_file)
    
    print(f"Data uploaded to AWS instance {instance_id}")


def _start_training_aws(instance_id, iterations):
    """Start training on AWS instance"""
    import boto3
    
    # Load instance info
    with open('cloud_instance_info.json', 'r') as f:
        instance_info = json.load(f)
    
    # Get instance public IP
    ec2 = boto3.client('ec2', region_name=instance_info['region'])
    response = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    # Start training
    key_file = instance_info['key_file']
    ssh_cmd = (
        f"ssh -i {key_file} -o StrictHostKeyChecking=no ubuntu@{public_ip} "
        f"'cd ~/cinematic-render && "
        f"python train.py --source_path=./scene --model_path=./model --iterations={iterations} > training.log 2>&1 &'"
    )
    
    subprocess.run(ssh_cmd, shell=True, check=True)
    print(f"Training started on AWS instance {instance_id}")


def _download_results_aws(instance_id, model_output_dir, compressed_output_dir):
    """Download results from AWS instance"""
    import boto3
    
    # Load instance info
    with open('cloud_instance_info.json', 'r') as f:
        instance_info = json.load(f)
    
    # Get instance public IP
    ec2 = boto3.client('ec2', region_name=instance_info['region'])
    response = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    key_file = instance_info['key_file']
    
    # Check if training is complete
    ssh_cmd = f"ssh -i {key_file} -o StrictHostKeyChecking=no ubuntu@{public_ip} 'ps aux | grep train.py | grep -v grep'"
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("Training is still running. Waiting for completion...")
        # Implement waiting logic here
        # For now, we'll just assume it's done
    
    # Create local directories
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(compressed_output_dir, exist_ok=True)
    
    # Download model results
    scp_cmd = f"scp -i {key_file} -r ubuntu@{public_ip}:~/cinematic-render/model/* {model_output_dir}/"
    subprocess.run(scp_cmd, shell=True, check=True)
    
    # Download compressed results
    scp_cmd = f"scp -i {key_file} -r ubuntu@{public_ip}:~/cinematic-render/compressed/* {compressed_output_dir}/"
    subprocess.run(scp_cmd, shell=True, check=True)
    
    print(f"Results downloaded from AWS instance {instance_id}")


# ------------------- GCP Implementation -------------------

def _setup_gcp_instance(instance_type, region, startup_script):
    """Set up a GCP Compute Engine instance with GPU"""
    # This would be implemented for GCP
    # Placeholder for demonstration purposes
    print("GCP implementation would set up a Compute Engine instance here")
    return "gcp-instance-id-placeholder"


def _upload_data_gcp(instance_id, data_dir, iterations):
    """Upload data to GCP instance"""
    # This would be implemented for GCP
    # Placeholder for demonstration purposes
    print("GCP implementation would upload data here")


def _start_training_gcp(instance_id, iterations):
    """Start training on GCP instance"""
    # This would be implemented for GCP
    # Placeholder for demonstration purposes
    print("GCP implementation would start training here")


def _download_results_gcp(instance_id, model_output_dir, compressed_output_dir):
    """Download results from GCP instance"""
    # This would be implemented for GCP
    # Placeholder for demonstration purposes
    print("GCP implementation would download results here")


# ------------------- Azure Implementation -------------------

def _setup_azure_instance(instance_type, region, startup_script):
    """Set up an Azure VM with GPU"""
    # This would be implemented for Azure
    # Placeholder for demonstration purposes
    print("Azure implementation would set up a VM here")
    return "azure-instance-id-placeholder"


def _upload_data_azure(instance_id, data_dir, iterations):
    """Upload data to Azure instance"""
    # This would be implemented for Azure
    # Placeholder for demonstration purposes
    print("Azure implementation would upload data here")


def _start_training_azure(instance_id, iterations):
    """Start training on Azure instance"""
    # This would be implemented for Azure
    # Placeholder for demonstration purposes
    print("Azure implementation would start training here")


def _download_results_azure(instance_id, model_output_dir, compressed_output_dir):
    """Download results from Azure instance"""
    # This would be implemented for Azure
    # Placeholder for demonstration purposes
    print("Azure implementation would download results here")


# ------------------- Helper Functions -------------------

def _generate_startup_script(provider):
    """Generate startup script for cloud instance"""
    if provider == 'aws':
        script = """#!/bin/bash
# Install required packages
pip install torch-scatter plyfile

# Clone repositories
git clone https://github.com/mmoritz2/Cinematic-Render-CT.git
cd Cinematic-Render-CT
git submodule update --init --recursive

# Prepare directories
mkdir -p ~/cinematic-render/scene
mkdir -p ~/cinematic-render/model
mkdir -p ~/cinematic-render/compressed

# Copy necessary scripts
cp -r cinematic-gaussians/* ~/cinematic-render/
cp ct-cinematic-viewer/train_compress.py ~/cinematic-render/

# Signal that setup is complete
touch ~/setup_complete
"""
    elif provider == 'gcp':
        # Adjust for GCP
        script = """#!/bin/bash
# GCP-specific setup script would go here
"""
    elif provider == 'azure':
        # Adjust for Azure
        script = """#!/bin/bash
# Azure-specific setup script would go here
"""
    else:
        script = ""
    
    return script


def _wait_for_instance_ready(provider, instance_id, region):
    """Wait for cloud instance to be ready"""
    if provider == 'aws':
        import boto3
        
        ec2 = boto3.client('ec2', region_name=region)
        
        print("Waiting for instance to be running...")
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Wait for status checks to pass
        print("Waiting for instance status checks...")
        waiter = ec2.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get instance details
        response = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        
        print(f"Instance is ready! Public IP: {public_ip}")
        
        # Load instance info
        with open('cloud_instance_info.json', 'r') as f:
            instance_info = json.load(f)
        
        # Update with public IP
        instance_info['public_ip'] = public_ip
        
        with open('cloud_instance_info.json', 'w') as f:
            json.dump(instance_info, f)
        
        # Wait for the startup script to complete
        key_file = instance_info['key_file']
        
        # Wait for SSH to be available
        print("Waiting for SSH to be available...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                ssh_cmd = f"ssh -i {key_file} -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@{public_ip} 'echo SSH is available'"
                result = subprocess.run(ssh_cmd, shell=True, capture_output=True)
                if result.returncode == 0:
                    print("SSH is available!")
                    break
            except Exception:
                pass
            
            print(f"Attempt {attempt+1}/{max_attempts}. Retrying in 10 seconds...")
            time.sleep(10)
        
        # Wait for setup to complete
        print("Waiting for instance setup to complete...")
        max_attempts = 60
        for attempt in range(max_attempts):
            try:
                ssh_cmd = f"ssh -i {key_file} -o StrictHostKeyChecking=no ubuntu@{public_ip} 'test -f ~/setup_complete && echo Setup complete'"
                result = subprocess.run(ssh_cmd, shell=True, capture_output=True)
                if result.returncode == 0 and b"Setup complete" in result.stdout:
                    print("Instance setup is complete!")
                    break
            except Exception:
                pass
            
            print(f"Attempt {attempt+1}/{max_attempts}. Retrying in 10 seconds...")
            time.sleep(10)
    
    elif provider == 'gcp':
        # Implement GCP-specific waiting logic
        print("GCP implementation would wait for instance readiness here")
        time.sleep(2)  # Placeholder
    
    elif provider == 'azure':
        # Implement Azure-specific waiting logic
        print("Azure implementation would wait for instance readiness here")
        time.sleep(2)  # Placeholder


def _cleanup_cloud_resources(provider, instance_id):
    """Clean up cloud resources after use"""
    if provider == 'aws':
        import boto3
        
        # Load instance info
        with open('cloud_instance_info.json', 'r') as f:
            instance_info = json.load(f)
        
        # Create EC2 client
        ec2 = boto3.client('ec2', region_name=instance_info['region'])
        
        # Terminate instance
        print(f"Terminating AWS instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        
        # Wait for termination
        waiter = ec2.get_waiter('instance_terminated')
        waiter.wait(InstanceIds=[instance_id])
        
        # Delete security group
        print("Deleting security group...")
        try:
            ec2.delete_security_group(GroupId=instance_info['security_group_id'])
        except Exception as e:
            print(f"Warning: Could not delete security group: {e}")
        
        # Delete key pair
        print("Deleting key pair...")
        ec2.delete_key_pair(KeyName=instance_info['key_name'])
        
        # Remove local key file
        if os.path.exists(instance_info['key_file']):
            os.unlink(instance_info['key_file'])
        
        # Remove instance info file
        if os.path.exists('cloud_instance_info.json'):
            os.unlink('cloud_instance_info.json')
        
        print("AWS resources cleaned up successfully")
    
    elif provider == 'gcp':
        # Implement GCP cleanup
        print("GCP implementation would clean up resources here")
    
    elif provider == 'azure':
        # Implement Azure cleanup
        print("Azure implementation would clean up resources here") 