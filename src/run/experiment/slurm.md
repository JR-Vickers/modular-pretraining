## Basic Setup

1. If John haven’t already sent you a sign up link, then post in **#fellows-tech-support-chatter** asking for someone to generate one. The org is called `Anthropic Safety Research`
2. This is [runpod's guide](https://www.notion.so/Creating-an-Account-and-SSH-into-Cluster-292ff732fc3480d0a39ee6a78db70f82?pvs=21) on how to create keys so you can ssh to our cluster and your own pods.
3. Add your public SSH key to RunPod UI and send it to Eugene in **#ext-fellows-runpod**
    1. Make sure you have the correct org chosen in the top left and navigate to https://www.runpod.io/console/pods → Settings
    2. Paste the *public* key you created (ending with .pub) in the “SSH Public Keys” box in the RunPod web UI
    3. ⚠️ DO NOT COPY ANY PRIVATE KEYS FROM YOUR LOCAL COMPUTER TO RUNPOD ⚠️

## Using the compute cluster

<aside>
💡

Note: RunPod’s guides are here [Slurm Wiki For Anthropic](https://www.notion.so/Slurm-Wiki-For-Anthropic-292ff732fc34807e94e2cea78eebfea6?pvs=21) 

We also have some troubleshooting in [Cluster Troubleshooting](https://www.notion.so/Cluster-Troubleshooting-2aaca43b7eec804fa897ed92c39f2565?pvs=21) 

Use the [RUNPOD_INFRASTRUCTURE_GUIDE.md](https://www.notion.so/Optional-Workflows-2a1ca43b7eec8095b332e2fe817270cb?pvs=21) made by Faizan! More cool workflows in [Optional Workflows](https://www.notion.so/Optional-Workflows-2a1ca43b7eec8095b332e2fe817270cb?pvs=21) 

</aside>

### What is it?

1. The cluster is set up in Anthropic Safety Research [here](https://console.runpod.io/cluster).
2. We have negotiated with RunPod to get 8xH200 nodes with a **35% discount ($2.30/hr)**.
3. We have 24 nodes (32 very soon) and we can expand as demand grows.
4. You have a 16 GPU limit for jobs that can’t be preempted (people can use unlimited preemptible jobs to soak up spare compute)
5. We have fast VAST storage mounted across nodes in `/workspace-vast`. We can easily increase this storage.
6. We have a dedicated RunPod engineer to help support us called Eugene Klitenik. Another engineer called Hailong Yang often helps and you can speak with them in **#ext-fellows-runpod**.
7. There are three different drives:
    1. Container disk (3TB on each node, fast): You will have your home dir `/home/<user>` here. Only recommended for temporary files. Not recommended to use for experiments since not mirrored across nodes.
    2. VAST storage (200TB, cross mounted, fast): use for all virtual environments, git repos, data and model checkpoints. If you do not put these on VAST, your jobs will fail when they land on other nodes. Please use `/workspace-vast/<user>` to store your data.
    3. Network drive (200TB, cross mounted, slow): used for cold storage if people have checkpoints or similar that they don’t need to access fast. It can be found mounted at `/workspace`.

There is lots of information here but we recommend using the [RUNPOD_INFRASTRUCTURE_GUIDE.md](https://www.notion.so/Optional-Workflows-2a1ca43b7eec8095b332e2fe817270cb?pvs=21) made by Faizan which explains everything to Claude Code!

### Using the cluster

1. RunPod will tell you which node you should do remote development on (to spread the load). See their guide [here in their wiki](https://www.notion.so/Creating-an-Account-and-SSH-into-Cluster-292ff732fc3480d0a39ee6a78db70f82?pvs=21). Add the machine details to your ssh config (~/.ssh/config by default - replacing `<user>` with yours, `<hostname>` and `<port>`  with what is in the node detail page in the RunPod UI) to access the cluster easily by running `ssh cluster` or finding it in the remote ssh VSCode plugin.
    1. In the screenshot below the <hostname> is 198.145.108.6 and the <port> is 10400. This can change if RunPod update stuff so best to check rather than rely on these.
    2. (optional) See this [script](https://github.com/safety-research/safety-tooling/pull/138/files) made by Shane which automatically adds for you.

```jsx
Host cluster
    HostName <hostname>
    User <user>
    Port <port>
	  IdentityFile ~/.ssh/id_ed25519
```

![Screenshot 2025-11-06 at 20.20.50.png](attachment:29a9d1c4-c6bf-487a-b4d3-32d06a4c1571:Screenshot_2025-11-06_at_20.20.50.png)

1. You should use the node Eugene tells you to use as the main node to do remote development (so use the standard remote ssh via Cursor/VSCode)
2. We use Slurm (Simple Linux Utility for Resource Management) for GPU scheduling. Slurm allows us to schedule jobs across multiple users and efficiently utilise our GPUs.
    1. See how to submit jobs and use interactive sessions in RunPod's [slurm guide](https://www.notion.so/Using-Slurm-Cluster-292ff732fc3480ff80dbfc0d5c99a599?pvs=21)
    2. See how to monitor, control priority, useful aliases and how to run vllm here: [GPU Scheduling with Slurm](https://www.notion.so/GPU-Scheduling-with-Slurm-1b5ca43b7eec80259a35d68f43b71481?pvs=21)
    3. ⚠️ DO NOT EVER EXPORT CUDA_VISIBLE_DEVICES YOURSELF. SLURM DOES THIS FOR YOU ⚠️
        1. If you do it will cause slurm to land jobs on GPUs that might be utilised causing all jobs to drain into that slot and crash!
    4. Only use nodes other than the one assigned to your for remote dev via a Slurm interactive session.
3. Create your directory on VAST by running `mkdir /workspace-vast/$(whoami)` 
4. Make sure to export `export HF_HOME=/workspace-vast/pretrained_ckpts` so we share the same huggingface cache across the cohort. Putting in your dotfiles would make sense.
    1. You should set `HF_TOKEN_PATH` to `/workspace-vast/$(whoami)/.cache/huggingface/token` otherwise tokens will be shared from HF_HOME.
5. Create your first uv venv
    
    ```bash
    # create git, experiment and environment directories on VAST
    mkdir -p /workspace-vast/$(whoami)/git /workspace-vast/$(whoami)/exp /workspace-vast/$(whoami)/envs
    
    # install venv that works across nodes
    export UV_PYTHON_INSTALL_DIR=/workspace-vast/$(whoami)/.uv/python
    export UV_CACHE_DIR=/workspace-vast/$(whoami)/.cache/uv
    mkdir -p $UV_PYTHON_INSTALL_DIR $UV_CACHE_DIR
    cd /workspace-vast/$(whoami)/envs
    uv python install 3.11
    uv venv
    source .venv/bin/activate
    ```
    
6. Here are scripts for an example workflow
    1. Seoirse’s [script](https://github.com/seoirsem/dotfiles/blob/master/runpod/slurm_gpu_visual.sh) to view available GPUs across nodes (it also resizes nicely in your terminal!)
        
        ![image (11).png](attachment:04ae9cae-6174-42dc-8fb6-f65ee10db25c:image_(11).png)
        
    2. Example [script](https://github.com/safety-research/safety-examples/blob/b0d7956deb6a16953624387501d7a80c9dfd8de8/examples/slurm/setup_axolotl.sh) to setup axolotl venv
    3. Example [train.sh script](https://github.com/safety-research/safety-examples/blob/b0d7956deb6a16953624387501d7a80c9dfd8de8/examples/slurm/train.sh) to schedule job via slurm and ensure it uses the correct venv, secrets, logging dir etc.
7. Claude code is very good at Slurm so I recommend using it to learn fast!
    
    ```bash
    sudo apt install nodejs npm gh
    sudo npm install -g @anthropic-ai/claude-code
    ```
    

### Interactive jobs

- Interactive jobs (aka dev jobs or qlogins) are designed to be used for development.
- When you run an interactive job, you are ssh’d into the node and CUDA_VISIBLE_DEVICES is exported with the GPU(s) assigned to you by the queue.
- Now you can debug jobs easily without having to always wait for it to get through the queue.
- Please prefix your interactive jobs with `D_` (e.g. `D_johnh`) which will allow a cronjob to delete the job at midnight PT. This means other jobs in the queue will use the compute overnight.
    - Dev jobs should not be used for long running experiments!
- We have a separate “dev” partition (usually set to be 1/8th the size of the cluster) that is carved out so that dev jobs can always be scheduled (unless it is already full of dev jobs). See more about the dynamics of this [below](https://www.notion.so/RunPod-Slurm-Cluster-172ca43b7eec808cbf12d2175f4c5191?pvs=21). This stops people getting blocked if there are lots of high priority jobs which can’t be preempted like low priority jobs.
    - We can expand this partition to 2 or more nodes if needed. Just post about it in **#ext-fellows-runpod** if you think it would help.

### Remote development

- We recommend remote development on the cluster and find it is the fastest way to run experiments.
- You can remote ssh to your node using VSCode/Cursor.
- Then you can directly make changes to your git repository in the `/workspace-vast/<user>/git` folder and jobs on any node can use those changes.
- You can debug in an interactive job (e.g. on node1) that is open in one terminal while editting the code open in VSCode which is operating on another node.
    - Some workflows might benedit from VScode debugging. If so, you’ll need to remote ssh to the node which your interactive job provides you.

### Queue partitions and quality of service

- **Priority hierarchy**: dev QoS (300) > high QoS (200) > low QoS (100) determines scheduling order and preemption rights
- **Preemption only on low QoS** - low QoS preempted jobs are automatically re-queued and will restart when resources become available
- **Some nodes are reserved primarily for dev work** - low QoS jobs can backfill but will be kicked off when dev needs it
- **Dev jobs must be interactive** - users cannot submit batch scripts with dev QoS
    - When using interactive jobs to do development work, please name them with `D_` prefix. Using this prefix means that they will be automatically deleted at midnight PT. This is great since often people forget about their interactive jobs and it means other stuff can take its place overnight.
        - You can start an interactive job like so `srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --job-name=D_<user> --mem=32G --pty bash`
- Extra detail for those interested. This is all managed through quality of service tiers (QoS) and partitions. Both of these concepts have a priority system.
    - QoS
        - `dev` - priority 300, dev (interactive) jobs always take priority over anything else so people don’t get blocked, they can’t get preempted
        - `high` - priority 200, jobs that people do not want preempted
        - `low` - priority 100, jobs people don’t mind getting preempted
    - Partitions
        - `dev` - Dev or low QoS allowed.
        - `general` - High and low QoS allowed
        - `overflow` - All QoS allowed.
        - This means interactive jobs will hit the dev partition first before going to others. To do this users specify multiple partitions (e.g., `p dev,overflow`) so Slurm tries them in order of partition priority.

### Limits

- GPU limits (for fairness)
    - High QoS → 16 GPUs max
    - Low QoS → unlimited jobs (but can be preempted)
    - Dev QoS → 8 GPU max
- 7 day max job length but 24 hours unless you explicitly set. Dev jobs have 24 hour max length.

### Blocked on compute?

- Use the channel **#fellows-cluster-coordination** to plan with other fellows how you can slot your job in and learn if running jobs are finishing soon.
- Think about if you need your results now or if you can leave overnight to utilise compute then.
- Remember that interactive sessions (dev jobs) have a node especially carved out that high priority jobs can’t use.
- If you have extra sweeps to run that you don’t mind taking longer, use low priority QoS! They may get preempted but you’ll soak up any spare compute when it is available.
- If you’re still blocked, then start a temporary pod separate from the cluster and move back to the cluster when there is capacity again.
- If we are always at capacity, it might make sense to scale up, please DM John to discuss.

### Footguns

- Exporting CUDA_VISIBLE_DEVICES and running stuff off queue -> this makes all jobs in the queue end up crashing as they take that slot
    - **Solution**: hopefully onboarding is clear not to do this! No other mitigation right now.
- Filling up the storage with e.g. model checkpoints
    - **Solution**: RunPod support will alert and help cleanup
- Accidentally using too much cpu on a node which crashes everything (including VSCode development)
    - **Solution**: Hasn’t been an issue yet but please let me know if you notice this happening.
- Accidentally mass deleting people's data on VAST
    - **Solution**: RunPod have now rolled out user’s owning data on VAST and we also have a backup on the network storage.
- Filling up the queue and there not being any GPUs available for dev
    - **Solution**: We have a separate partition for interactive jobs so people shouldn’t get blocked unless it is over subscribed.

### More detail on cluster partition logic and how interactive sessions work

Explaining the cluster partition logic and how interactive sessions work. This will assume we have 8 nodes in the cluster (node0-node7).

- Two type of jobs:
    - Interactive session (aka dev job) -> uses `srun [args] --pty bash`
        - these are jobs where you need to grab GPU(s) for debugging or derisking lots of small experiments where it helps to have a persistent GPU assigned to you. When you request a dev job you will be SSH-ed to the node and CUDA_VISIBLE_DEVICES will already be exported for you.
            - Please use "D_<user>" for these jobs. They are intended for testing jobs during the day and not for long running experiments. If you need to run something overnight, kill your dev job and you a high or low prio job.
            - D_user jobs will get killed at midnight PT (this is 8am UK time which I think probably still works ok too?)
    - Normal jobs - uses `sbatch [args]`
        - a job you have fully debugged and you want to send to the queue and leave to run in the background once it gets a slot
        - usually long running
        - you can queue up many of these and do big sweeps easily (it will parallelise across available compute)
        - Note: When running multiple similar jobs (e.g., one per condition, hyperparameter, or data split), always use job arrays (--array=0-N) rather than submitting individual jobs in a loop. Define your conditions in a bash array and index with ${SLURM_ARRAY_TASK_ID}. This reduces scheduler overhead and avoids NODE_FAILs that sometimes happen when multiple jobs initialize on the same node simultaneously. (See the https://slurm.schedmd.com/job_array.html for details on job arrays.)
- We have 3 main **quality of service** (qos) tiers
    - **Dev** - top priority since debugging/derisking jobs should not be blocked
    - **High** - these are normal jobs that won't be preempted. You should use this for experiments where you've already debugged your scripts.
    - **Low** - these are normal jobs that can be preempted. You should use these for experiments where you'd like to utilise the spare compute but don't mind if results take longer. If you use this you will need to make sure your job can carry on where it left off killed and restarted (e.g. save out checkpoints and ensure they are picked up on rerun).
        - There will be a 3 min grace period where you can trap on SIGTERM to save out checkpoints too - we can increase if needed
- Since dev jobs are important to maintain capacity for (since people need them quickly to start debugging stuff), we use **separate partitions** of nodes to help with this.
    - **General partition** -> this is node 0-6. This is primarily for normal jobs
    - **Dev partition** -> this is node 7. It is separated so people can get GPUs quickly for dev.
        - Note: If you try to submit a normal job (with sbatch) here it will get rejected
- Then there are 2 things we still want:
    - 1) We want to stop high prio going on dev (since they're not pre-emptible) BUT we want to allow low prio jobs to fill up the gaps
    - 2) We want dev jobs to land on node 7 first but then still allow dev jobs to use the rest of the cluster if there is capacity
    - So this is where the **overflow partition** comes in. This allows dev jobs to overflow into general. And it allows low priority to overflow into dev.
- So in practise this means high prio jobs can't go on node 7 but low prio / dev jobs can go anywhere (its just dev jobs will prioritise filling up node 7 first)

Quick reference for commands:

- If you want an interactive session use `srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --job-name=D_<user> --pty bash`
- If you want a high prio job (that won't get preempted) use `sbatch -p general --qos=high job.sh`
- If you want a low prio job to fill excess compute but can get preempted use `sbatch -p general,overflow --qos=low job.sh`

### Miscellaneous

- Sometimes you will observe the time for loading models (e.g. huggingface transformers) into VRAM varying wildly, e.g. by a factor of 5x. This is likely due to the linux page cache caching the weights into RAM for subsequent loads. This caching happens per-node the first time the model is loaded on that node, meaning that subsequent loads can be significantly faster.
    - This underscores the value of interactive slurm jobs — keeping your work on the same node ensures you benefit from these types of caching!

## Deploy your own pod

<aside>
💡

If the cluster isn’t working for you, please talk to John about why and discuss options (we have good support from RunPod so would like to fix any pain points you have). 

Depolying your own pods is an alternative. We have 10% discount on all on demand prices. However, Hyperbolic is still cheaper so check them out: [Hyperbolic](https://www.notion.so/Hyperbolic-2a1ca43b7eec800085bbe646d144d031?pvs=21) 

</aside>

### Important norms

- Make sure pods include your name in the title you provide so we know who it belongs to. Also include the project name and how long you expect to have it running for. **If you to not include your name in the pod name, we may shut it down when doing spot checks on usage.** Example names include:
    - `john-hughes-bon-jailbreaking-5days`
    - `john-hughes-devbox-always-on`
- Please be aware of how much it costs to leave a pod running
    - 1 x H100 PCIe costs around $2.70/hr which means it costs just shy of $2k to run all month. This is well worth it if you are not spending lots on other compute or constantly debugging/running experiments on a GPU each day. However, if you think it will be unused for more than a 1-2weeks (and it won’t be painful to setup the environment again), we recommend shutting it down to save on your monthly budget.
- It is worth investing time in a script that quickly deploys everything you need on a fresh pod. Therefore, the barrier to entry of starting and stopping pods is less. See an example [here](https://github.com/jplhughes/dotfiles/blob/master/runpod/runpod_setup.sh).
- Always use shared network drives when using RunPod (and share the same one within your project). You can save experiment artefacts, data, and model checkpoints that can be accessible by others easily in their pods too.
    - Only things you need to run fast should be kept off the network drive (e.g. the git repo, venv and model cache)
    - It only costs $50 per TB per month. We recommend you start with 500GB-1TB and increase as you need more (you can increase but not decrease volume sizes). The current limit for network drives is 4TB.
- If you burst too many GPUs, first figure out how long you can keep them up before you hit your monthly budget. We are still working with RunPod to get user budgets but right now we rely on collaborators being aware of their expenditure. If you need more money for compute or expect to need to run an expensive experiment for a paper, let John know (we can probably reassign budget from elsewhere to make things work for you).

### Getting started

1. Deploy a pod
    1. Make sure you have the correct org chosen in the top right and navigate to https://www.runpod.io/console/pods and click deploy
    2. add a network volume
        1. choose a data center that has good availability of the GPUs you want (US-KS-2 seems pretty good usually)
        2. Name it with your name or project name
        3. Choose 500GB-1TB to begin with (since you can always increase but not decrease size)
    3. Choose the GPU you want and give the pod a name that clearly states your name and how long you expect to run it for (e.g. 1week or always on).
        1. E.g. `johnh-adversarial-training-1-week`
        2. If it is a devbox you will keep on use e.g. `johnh-devbox-always-on`
        3. Do a quick calculation that running the chosen number of GPUs is within your budget for the month. If you need more, speak to Henry/John.
            1. There is no spending per user tracking so we trust you to keep tabs on your spending manually
    4. (optional) you can add the VAST storage that the cluster uses by following these steps: [https://www.notion.so/runpod/Creating-a-individual-Pod-to-connect-to-vast-2b1ff732fc3480bda3f4c7e6a1476556](https://www.notion.so/Creating-a-individual-Pod-to-connect-to-vast-2b1ff732fc3480bda3f4c7e6a1476556?pvs=21)
2. Connect to a pod
    1. Find your pod and click connect to see the instructions
    2. It should give you a command for **SSH over exposed TCP**
    3. Add these details to your .ssh/config (make sure you have created an ssh key with `ssh-keygen -t ed25519 -C "your_email@example.com"`)
        
        ```python
        Host runpod
            HostName XXX.XX.XXX.XX
            User root
            Port <port>
            IdentityFile ~/.ssh/id_ed25519
        ```
        
    4. Now you can ssh to your pod with `ssh runpod` 
    5. You will now be able to remote ssh via Cursor or VSCode and this runpod option will appear in the dropdown to choose from
3. Setup the pod for development
    1. We recommend you write your own setup script like this https://github.com/jplhughes/dotfiles/blob/master/runpod/runpod_setup.sh
    2. This example script installs basic linux tools, creates a virtual env and creates an ssh key to add to your github so you can pull dotfiles (see [Workflow Tips](https://www.notion.so/Workflow-Tips-169ca43b7eec80e3acd3eaa4d22ec3e9?pvs=21) for info on dotfiles)
    3. You can run quickly once you’re on the pod with: `curl -s https://raw.githubusercontent.com/jplhughes/dotfiles/master/runpod/runpod_setup.sh | bash` 
    4. It won’t change the default shell to zsh until you close and reopen the connection
4. Git clone your repo locally and keep all data/models/results on your shared volume in /workspace
5. **Important**: since you’re going to add secrets like API keys and GitHub tokens the machine, edit `~/.ssh/authorized_keys` to only contain your key (since by default everyone in the RunPod org has access to your pod). This reduces the risk of an attacker being able to access all pods if one user is compromised.

### Moving data between runpod pods

RunPod have a guide for moving data between data centers here:

[Moving_Data_Between_Data_Centers.pdf](attachment:55ee0ab5-4438-4120-b093-6a166292ba23:Moving_Data_Between_Data_Centers.pdf)

[GPU Scheduling with Slurm](https://www.notion.so/GPU-Scheduling-with-Slurm-1b5ca43b7eec80259a35d68f43b71481?pvs=21)

[Optional Workflows](https://www.notion.so/Optional-Workflows-2a1ca43b7eec8095b332e2fe817270cb?pvs=21)

[Cluster Troubleshooting](https://www.notion.so/Cluster-Troubleshooting-2aaca43b7eec804fa897ed92c39f2565?pvs=21)

Example train.sh:

```sh
#!/bin/bash
user=$(whoami)
timestamp=$(date +%Y%m%d_%H%M%S)
work_dir=/workspace-vast/$user/exp/$(date +%Y%m%d)_slurm_test
venv_dir=/workspace-vast/$user/git/venvs/venv_axolotl
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config=$SCRIPT_DIR/meta-llama-8b-fsdp.yaml
env_file=/workspace-vast/$user/.env
huggingface_home=/workspace-vast/pretrained_ckpts

# GPU and memory configuration
num_gpus=4
mem_per_gpu=200  # GB per GPU, leaves ~400GB overhead on 2TB nodes
total_mem=$((num_gpus * mem_per_gpu))

# Ensure the experiment directory and logs directory exist
mkdir -p $work_dir/logs

# ensure HF_TOKEN and WANDB_TOKEN are in the .env file
if ! grep -q "HF_TOKEN" $env_file; then
  echo "Error: HF_TOKEN and/or WANDB_TOKEN not found in $env_file"
  exit 1
fi

cp $env_file $work_dir/.env

cat <<EOL > $work_dir/train.qsh
#!/bin/bash
#SBATCH --job-name=8B_ft
#SBATCH --output=$work_dir/logs/8B_ft_${timestamp}.out
#SBATCH --error=$work_dir/logs/8B_ft_${timestamp}.err
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --partition=high
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${total_mem}G
#SBATCH --chdir=$work_dir


export \$(grep -v '^#' $work_dir/.env | xargs)
export HF_HOME=$huggingface_home
export ACCELERATE_MAIN_PROCESS_PORT=0
source $venv_dir/bin/activate
axolotl preprocess $config
axolotl train $config
EOL

# submit job
sbatch $work_dir/train.qsh

echo "To see the queue, run:"
echo "watch squeue"

# echo the log file
echo "To view the log file, run:"
echo "tail -f $work_dir/logs/8B_ft_${timestamp}.out"
```


----


# Creating an Account and SSH into Cluster

This is a guide to get SSH access to the Slurm Cluster

## 1. How to Generate Your SSH Key

### Step 1: Open Terminal/Command Prompt

```markdown
- **Mac**: Open Terminal (Applications > Utilities > Terminal)
- **Linux**: Open your terminal application
- **Windows**: Open PowerShell or Windows Terminal
```

### Step 2: Generate the SSH Key

```markdown
Run this command from any directory (works from Desktop, Downloads, etc.):

```bash
ssh-keygen -t ed25519 -C "your.email@company.com"
```
```

Replace `your.email@company.com` with your actual email address.

### Step 3: Follow Prompts

You’ll probably see things similar to this:

(Feel free to rename the SSH file if you would like, just be aware of the instructions)

```markdown
```
Generating public/private ed25519 key pair.
Enter file in which to save the key (/Users/yourname/.ssh/id_ed25519): [press Enter]
Enter passphrase (empty for no passphrase): [enter a strong passphrase]
Enter same passphrase again: [repeat the same passphrase]

Your identification has been saved in /Users/yourname/.ssh/id_ed25519
Your public key has been saved in /Users/yourname/.ssh/id_ed25519.pub
The key fingerprint is:
SHA256:abc123xyz... your.email@company.com
```
```

**Important Notes:**

- Press Enter to accept the default file location
- Two files are created:
  - `id_ed25519` - Your **private key** (NEVER share this)
  - `id_ed25519.pub` - Your **public key** (this is what you'll send to us)

### Step 4: Get Your Public Key

Run this command to display your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

You should see something such as:

```bash
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPqq86fICaIPzhJ3jdxbVEN5lp96NDuJ6w23cdm7DWus your.email@company.com
```

### Send the Following Information

Send a message to Runpod with:
1. **Your desired/approved username** (e.g., "jsmith", "alice", "bob")

- Ask your organization if any rules you need to follow or if you just need to notify them of your username

2. **The complete public key** (the entire line from the command above)
   - It should start with `ssh-ed25519` or `ssh-rsa`
   - Include everything on the line, including your email at the end

## Step 5 Connecting:

Depending on your username (usually first name + last-name initial), you can use the table below to pick your dedicated login node.

If your assigned node isn’t working for any reason, feel free to use any node below/above it in the list.

- A–B: node-25
- C: node-2, node-4, node-5
- D–H: node-6, node-7
- I–J: node-8, node-9
- K: node-10
- L: node-11
- M: node-12
- N: node-13
- O–R: node-14, node-15
- S: node-16, node-17, node-18, node-19, node-20
- T–W: node-21, node-22
- X–Z: node-23, node-24

For example, my username is **hailongy,** it starts with H, so my assigned node would be **`node-6 or 7`**.

```bash
# node-0 - LOGIN DISABLED
~~ssh you_username@198.145.108.13 -p 15215 -i ~/.ssh/id_ed25519~~

# node-1
ssh you_username@198.145.108.32 -p 16281 -i ~/.ssh/id_ed25519

# node-2
ssh you_username@198.145.108.21 -p 17096 -i ~/.ssh/id_ed25519

# node-3
ssh you_username@198.145.108.25 -p 11251 -i ~/.ssh/id_ed25519

# node-4
ssh you_username@198.145.108.11 -p 10057 -i ~/.ssh/id_ed25519

# node-5
ssh you_username@198.145.108.38 -p 12488 -i ~/.ssh/id_ed25519

# node-6
ssh you_username@198.145.108.46 -p 18625 -i ~/.ssh/id_ed25519

# node-7
ssh you_username@198.145.108.10 -p 12370 -i ~/.ssh/id_ed25519

# node-8
ssh you_username@198.145.108.51 -p 13219 -i ~/.ssh/id_ed25519

# node-9
ssh you_username@198.145.108.47 -p 17040 -i ~/.ssh/id_ed25519

# node-10
ssh you_username@198.145.108.15 -p 17778 -i ~/.ssh/id_ed25519

# node-11,12,13
ssh you_username@198.145.108.41 -p 17228 -i ~/.ssh/id_ed25519
ssh you_username@198.145.108.62 -p 13450 -i ~/.ssh/id_ed25519
ssh you_username@198.145.108.29 -p 13467 -i ~/.ssh/id_ed25519

# node-14
ssh you_username@198.145.108.14 -p 18818 -i ~/.ssh/id_ed25519

# node-15
ssh you_username@198.145.108.18 -p 19105 -i ~/.ssh/id_ed25519

# node-16
ssh you_username@198.145.108.45 -p 15781 -i ~/.ssh/id_ed25519

# node-17
ssh you_username@198.145.108.67 -p 14638 -i ~/.ssh/id_ed25519

# node-18
ssh you_username@198.145.108.44 -p 10028 -i ~/.ssh/id_ed25519

# node-19
ssh you_username@198.145.108.50 -p 13075 -i ~/.ssh/id_ed25519

# node-20
ssh you_username@198.145.108.22 -p 17941 -i ~/.ssh/id_ed25519

# node-21
ssh you_username@198.145.108.37 -p 17285 -i ~/.ssh/id_ed25519

# node-22
ssh you_username@198.145.108.31 -p 17494 -i ~/.ssh/id_ed25519

# node-23
ssh you_username@198.145.108.61 -p 19934 -i ~/.ssh/id_ed25519

# node-24
ssh you_username@198.145.108.23 -p 14413 -i ~/.ssh/id_ed25519

# node-25
ssh you_username@198.145.108.6 -p 16240 -i ~/.ssh/id_ed25519

# node-26
ssh you_username@198.145.108.64 -p 16948 -i ~/.ssh/id_ed25519

# node-27
ssh you_username@198.145.108.26 -p 12680 -i ~/.ssh/id_ed25519

# node-28
ssh you_username@198.145.108.24 -p 14022 -i ~/.ssh/id_ed25519

# node-29
ssh you_username@198.145.108.17 -p 15377 -i ~/.ssh/id_ed25519

# node-30
ssh you_username@198.145.108.42 -p 18556 -i ~/.ssh/id_ed25519

# node-31
ssh you_username@198.145.108.65 -p 15776 -i ~/.ssh/id_ed25519
```

## Copy Data To Cluster

scp -P <port num>  <src> [user@](mailto:eugene@198.145.108.21)<node ip>:<dest>

Put this in your ssh config at ~/.ssh/config.

Change User

```
  Host node*
      User chrisv
      ServerAliveInterval 60
      ServerAliveCountMax 10
      ConnectionAttempts 3
      ConnectTimeout 30
      Compression yes
      TCPKeepAlive yes
      IdentityFile ~/.ssh/id_ed25519

  Host node-8
      HostName 198.145.108.51
      Port 15115

  Host node-0
      ProxyJump node-8
  Host node-1
      ProxyJump node-8
  Host node-2
      ProxyJump node-8
  Host node-3
      ProxyJump node-8
  Host node-4
      ProxyJump node-8
  Host node-5
      ProxyJump node-8
  Host node-6
      ProxyJump node-8
  Host node-7
      ProxyJump node-8
  Host node-9
      ProxyJump node-8
  Host node-10
      ProxyJump node-8
  Host node-11
      ProxyJump node-8
  Host node-12
      ProxyJump node-8
  Host node-13
      ProxyJump node-8
  Host node-14
      ProxyJump node-8
  Host node-15
      ProxyJump node-8
  Host node-16
      ProxyJump node-8
  Host node-17
      ProxyJump node-8
  Host node-18
      ProxyJump node-8
  Host node-19
      ProxyJump node-8
  Host node-20
      ProxyJump node-8
  Host node-21
      ProxyJump node-8
  Host node-22
      ProxyJump node-8
  Host node-23
      ProxyJump node-8
```