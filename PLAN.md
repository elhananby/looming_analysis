We will use this repository to create analysis scripts and notebooks, which will allows the user to analyse `.braidz` files.
`.braidz` is a file format created by the `braid` tracking program. In essence, it is a zipfile, which contains several compressed and non-compressed csv files, the most important are:
- `kalman_estimates.csv.gz` - with columns `obj_id, frame,timestamp, x, y, z, xvel, yvel, zvel` (and a few other less important ones).
- `stim.csv` - where each row contains a copy of a row from the `kalman_estimates` and additional columns with the stimuli information.

Your task is to write a Jupyter notebook where the user can load one or more `.braidz` files, and then perform analysis on them.
The main analysis will be getting the average angular velocity response to different stimuli presentations, where in our case all the stimuli will be looming (expanding circle) stimuli, with specific parameters. They can have different combinations of postion, speed, size, etc; we want to group them according to user specified parameters (for example, taking all the looms that approached the fly at -90), and getting an analysis and plotting of the average angula velocity response.
Usually, angular velocity is calculate by:
```Python
theta = np.arctan2(yvel, xvel)
theta_unwrap = np.unwrap(theta)
angular_velocity = np.gradient(theta_unwrap, 0.01) # assume always a recording framerate of 100hz
```
Where the different variables may need to be smoothed out before or after (`xvel`, `yvel` are already slightly smoothed kalman estimates).
Each row in `stim.csv` indicates the beginning of a stimulus presentation. we want to see how the fly (identified by `obj_id`) responded to the stimulus.

Rules:
- Use uv for all package and source managment, as well as for running python commands.
- Use ruff for linting and formatting.
- Use typing and write explanatory docstrings.
- Commit often.
- Do not assume anything - ask clarifying questions if needed.

We are starting from a completely empty repo. I have copied a couple of `.braidz` file in the `data` folder which you can use for analysis. 
Initially, write everything inside a `Jupyter Notebook` file which I can then run and test. Later we may split it into different modules for organization.
