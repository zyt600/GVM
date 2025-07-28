# Setup

To use the custom interceptor, run this before running your script:

```
./build.sh && source set_env.sh
```

# Utilizing the Scheduler

To run an online process, you can add a hint to your command like so:

```
ONLINE_KERNEL=1 <script>
```

To run an offline process, you can add a hint to your command like so:

```
ONLINE_KERNEL=0 <script>
```