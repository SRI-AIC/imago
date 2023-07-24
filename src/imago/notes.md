## 210731

Specifier for spaces.Box, allowing each channel to represent a specific type of target 
(categorical, regression, sparse-regression).

TODO: Extend to spaces.Dict, e.g., from Cameleon,

        self.observation_space = spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(width, height, 3),
                                    dtype='uint8')
        self.observation_space = spaces.Dict({
            'image': self.observation_space})

TODO: Include code that converts a spaces.Dict into a batched observation.  We may
have to consider whether or not to have separate encoding piplines for them, as opposed
to attempting to integrate them all together.