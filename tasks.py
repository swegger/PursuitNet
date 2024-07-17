import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from abc import abstractmethod
from pursuitnet.nets.losses import PositionLoss, VelocityLoss, L2xDxActivationLoss, L2xDxRegularizer, CompoundedLoss
from copy import deepcopy
from typing import Union

class Task(tf.keras.utils.Sequence):
    """Base class for tasks.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        initial_joint_state: `Tensor` or `numpy.ndarray`, the desired initial joint states for the task, if a single set
            of pre-defined initial joint states is desired. If `None`, the initial joint states will be drawn from the
            :class:`motornet.nets.layers.Network.get_initial_state` method at each call of :meth:`generate`.

            This parameter will be ignored on :meth:`generate` calls where a `joint_state` is provided as input
            argument.
        name: `String`, the name of the task object instance.
    """
    def __init__(self, dt, name: str = 'Task'):
        self.__name__ = name
        self.dt = dt
        self.training_batch_size = 32
        self.training_n_timesteps = 100
        self.delay_range = [0, 0]

        self.initial_state_names = [
            'targetVelocity',
        ]
        self.output_names = [
            'targetVelocity',
        ]

        self.losses = {name: None for name in self.output_names}
        self.loss_names = {name: name for name in self.output_names}
        self.loss_weights = {name: 0. for name in self.output_names}
        self._losses = {name: [] for name in self.output_names}
        self._loss_names = {name: [] for name in self.output_names}
        self._loss_weights = {name: [] for name in self.output_names}

        self.convert_to_tensor = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))

    def add_loss(self, assigned_output, loss, loss_weight=1., name=None):
        """Add a loss to optimize during training.

        Args:
            assigned_output: `String`, the output state that the loss will be applied to. This should correspond to
                an output name from the :class:`Network` object instance passed at initialization. The output names
                can be retrieved via the :attr:`motornet.nets.layers.Network.output_names` attribute.
            loss: :class:`tensorflow.python.keras.losses.Loss` object class or subclass. `Loss`
                subclasses specific to `MotorNet` are available in the :class:`motornet.nets.losses` module.
            loss_weight: `Float`, the weight of the loss when all contributing losses are added to the total loss.
            name: `String`, the name (label) to give to the loss object. This is used to print, plot, and
                save losses during training.

        Raises:
            ValueError: If the `assigned_output` passed does not match any network output name.
        """
        if assigned_output not in self.output_names:
            raise ValueError("The assigned output passed does not match any network output name.")

        is_compounded = True if self._losses[assigned_output] else False
        keep_default_name = False

        if name is not None:
            # if a name is given, overwrite the default name assigned at initialization
            self._loss_names[assigned_output].append(name)
        elif hasattr(loss, 'name'):
            # else if the loss object has a name attribute, then use that name instead
            self._loss_names[assigned_output].append(loss.name)
        else:
            keep_default_name = True
            self._loss_names[assigned_output].append('subloss_' + str(len(self._loss_names[assigned_output] + 1)))

        if is_compounded:
            self.loss_names[assigned_output] = assigned_output.replace(' ', '_') + '_compounded'
        elif keep_default_name is False:
            self.loss_names[assigned_output] = self._loss_names[assigned_output][0]

        self.loss_weights[assigned_output] = 1. if is_compounded else loss_weight
        self._loss_weights[assigned_output].append(loss_weight)

        self._losses[assigned_output].append(deepcopy(loss))
        if is_compounded:
            losses = self._losses[assigned_output]
            loss_weights = self._loss_weights[assigned_output]
            self.losses[assigned_output] = CompoundedLoss(losses=losses, loss_weights=loss_weights)
        else:
            self.losses[assigned_output] = self._losses[assigned_output][0]

    @abstractmethod
    def generate(self, batch_size, n_timesteps, validation: bool = False):
        """Generates inputs, targets, and initial states to be passed to the `model.fit` call.

        Args:
            batch_size: `Integer`, the batch size to use to create the inputs, targets, and initial states.
            n_timesteps: `Integer`, the number of timesteps to use to create the inputs and targets. Initial states do
                not require a time dimension.
            validation: `Boolean`, whether to generate trials for validation purposes or not (as opposed to training
                purposes). This is useful when one wants to test a network's performance in a set of trial types that
                are not the same as those used for training.

        Returns:
            - A `dictionary` to use as input to the model. Each value in the `dictionary` should be a `tensor` array.
              At the very least, the `dictionary` should contain a "inputs" key mapped to a `tensor` array, which will
              be passed as-is to the network's input layer. Additional keys will be passed and handled didderently
              depending on what the :class:`Network` passed at initialization does when it is called.
            - A `tensor` array of target values, that will be passed to all losses as the `y_true` input to compute
              loss values.
            - A `list` of initial state as `tensor` arrays, compatible with the :attr:`initial_joint_state` value set at
              initialization.
        """
        return

    def get_input_dim(self):
        """Gets the dimensionality of each value in the input `dictionary` produced by the :meth:`generate` method.

        Returns:
            A `dictionary` with keys corresponding to those of the input `dictionary` produced by the :meth:`generate`
            method, mapped to `lists` indicating the dimensionality (shape) of each value in the input `dictionary`.
        """

        [inputs, _] = self.generate(batch_size=1, n_timesteps=self.delay_range[-1]+1)

        def sort_shape(i):
            if tf.is_tensor(i):
                s = i.get_shape().as_list()
            elif isinstance(i, np.ndarray):
                s = i.shape
            else:
                raise TypeError("Can only take a tensor or numpy.ndarray as input.")
            return s[-1]

        if type(inputs) is dict:
            shape = {key: sort_shape(val) for key, val in inputs.items()}
        else:
            shape = inputs

        return shape

    def get_losses(self):
        """Gets the currently declared losses and their corresponding loss weight.

        Returns:
            - A `dictionary` containing loss objects.
            - A `dictionary` containing `float` values corresponding to each loss' weight.
        """
        return [self.losses, self.loss_weights]

    def print_losses(self):
        """Prints all currently declared losses in a readable format, including the default losses declared at
        initialization. This method prints the assigned output, loss object instance, loss weight and loss name of each
        loss. It also specifies if each loss is part of a compounded loss or not.
        """
        for key, val in self._losses.items():
            if val:
                for n, elem in enumerate(val):
                    title = "ASSIGNED OUTPUT: " + key
                    print(title)
                    print("-" * len(title))
                    print("loss function: ", elem)
                    print("loss weight:   ", self._loss_weights[key][n])
                    print("loss name:     ", self._loss_names[key][n])
                    if len(val) > 1:
                        print("Compounded:     YES")
                    else:
                        print("Compounded:     NO")
                    print("\n")

    def get_attributes(self):
        """Gets all non-callable attributes declared in the object instance, except for loss-related attributes.

        Returns:
            - A `list` of attribute names as `string` elements.
            - A `list` of attribute values.
        """
        blacklist = ['loss_weights', 'losses', 'loss_names']
        attributes = [
            a for a in dir(self)
            if not a.startswith('_') and not callable(getattr(self, a)) and not blacklist.__contains__(a)
        ]
        values = [getattr(self, a) for a in attributes]
        return attributes, values

    def print_attributes(self):
        """Prints all non-callable attributes declared in the object instance, except for loss-related attributes.
        To print loss-related attributes, see :meth:`print_losses`."""
        attributes = [a for a in dir(self) if not a.startswith('_') and not callable(getattr(self, a))]
        blacklist = ['loss_weights', 'losses', 'loss_names']

        for a in attributes:
            if not blacklist.__contains__(a):
                print(a + ": ", getattr(self, a))

        for elem in blacklist:
            print("\n" + elem + ":\n", getattr(self, elem))

    def set_training_params(self, batch_size, n_timesteps):
        """Sets default training parameters for the :meth:`generate` call. These will be overridden if the
        :meth:`generate` method is called with alternative values for these parameters.

        Args:
            batch_size: `Integer`, the batch size to use to create the inputs, targets, and initial states.
            n_timesteps: `Integer`, the number of timesteps to use to create the inputs and targets. Initial states do
                not require a time dimension.
        """
        self.training_batch_size = batch_size
        self.training_n_timesteps = n_timesteps
        # self.training_iterations = iterations

    def get_save_config(self):
        """Gets the task object's configuration as a `dictionary`.

        Returns:
            A `dictionary` containing the  parameters of the task's configuration. All parameters held as non-callbale
            attributes by the object instance will be included in the `dictionary`.
        """

        cfg = {'name': self.__name__}  # 'training_iterations': self.training_iterations,
        attributes, values = self.get_attributes()
        for attribute, value in zip(attributes, values):
            if isinstance(value, np.ndarray):
                print("WARNING: One of the attributes of the Task object whose configuration dictionary is being "
                      "fetched is a numpy array, which is not JSON serializable. This may result in an error when "
                      "trying to save the model containing this Task as a JSON file. This is likely to occur with a "
                      "custom Task subclass that includes a custom attribute saved as a numpy array. To avoid this, it "
                      "is recommended to ensure none of the attributes of the Task are numpy arrays.")
            cfg[attribute] = value

        # save all losses as a list of dictionaries, each containing the information for one contributing loss.
        losses = []
        for key, val in self._losses.items():
            if val:
                for n, elem in enumerate(val):
                    d = {"assigned output": key, "loss object": str(elem), "loss name": self._loss_names[key][n],
                         "loss weight": self._loss_weights[key][n]}
                    if len(val) > 1:
                        d["compounded"] = True
                    else:
                        d["compounded"] = False
                    losses.append(d)
        cfg["losses"] = losses

        return cfg

    def __getitem__(self, idx):
        [inputs, targets, init_states] = self.generate(
            batch_size=self.training_batch_size,
            n_timesteps=self.training_n_timesteps
        )
        return [inputs, init_states], targets

    # def __len__(self):
    #     return self.training_iterations

    def get_input_dict_layers(self):
        """Creates :class:`tensorflow.keras.layers.Input` layers to build the entrypoint layers of the network inputs.
        See the `tensorflow` documentation for more information about what :class:`tensorflow.keras.layers.Input`
        objects do. Below is an example code using the current method to create a model. See
        :meth:`get_initial_state_layers` for more information about how to create a set of input layers for initial
        states.

        .. code-block:: python

            import motornet as mn
            import tensorflow as tf

            plant = mn.plants.ReluPointMass24()
            network = mn.nets.layers.GRUNetwork(plant=plant, n_units=50)
            task = mn.tasks.CentreOutReach(network=network)

            rnn = tf.keras.layers.RNN(cell=network, return_sequences=True)

            inputs = task.get_input_dict_layers()
            state_i = task.get_initial_state_layers()
            state_f = rnn(inputs, initial_state=state_i)

        Returns:
            A `dictionary`, with the same keys as the ``inputs`` dictionary from the :meth:`generate` method. These keys
            are mapped onto :class:`tensorflow.keras.layers.Input` object instances with dimensionality corresponding to
            the inputs provided in the ``inputs`` dictionary from the :meth:`generate` method.
        """
        return {key: Input((None, val,), name=key) for key, val in self.get_input_dim().items()}

class SmoothPursuitDirectional(Task):
    """During training, the network will perform smooth pursuit starting from random locations. During validation, the network will perform
    smooth pursuit in uniformly distributed directions from a fixation position.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass. This is the network that will perform
            the task.
        name: `String`, the name of the task object instance.
        angular_step: `Float`, the angular distance (deg) between each centre-out reach during validation. For instance,
            if this is `45`, the `Task` object will ask the network to perform reaches in `8` directions equally
            distributed around the center position.
        catch_trial_perc: `Float`, the percentage of catch trials during training. A catch trial is a trial where no
            go-cue occurs, ensuring the network has to learn to wait for the go cue to actually occur without trying
            to "anticipate" the timing of the go-cue.
        reaching_distance: `Float`, the reaching distance (m) for each centre-out reach during validation.
        start_position: `List`, `tuple` or `numpy.ndarray`, indicating the start position around which the centre-out
            reaches will occur during validation. There should be as many elements as degrees of freedom in the plant.
            If `None`, the start position will be defined as the center of the joint space, based on the joint limits of
            the plant.
        deriv_weight: `Float`, the weight of the muscle activation's derivative contribution to the default muscle L2
            loss.
        motion_onset_range: Two-items `list`, `tuple` or `numpy.ndarray`, indicating the lower and upper range of the time
            window (sec) in which the go cue may be presented. The go cue timing is randomly drawn from a uniform
            distribution bounded by these values.
        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
            self,
            dt: float = 0.01,
            name: str = 'SmoothPursuitDirectional',
            angular_step: float = 45,
            catch_trial_perc: float = 50,
            speed: float = 0.1,
            start_position: Union[list, tuple, np.ndarray] = None,
            deriv_weight: float = 0.,
            motion_onset_range: Union[list, tuple, np.ndarray] = (0.05, 0.25),
            **kwargs
    ):

        super().__init__(dt=dt, name=name, **kwargs)

        self.angular_step = angular_step
        self.catch_trial_perc = catch_trial_perc
        self.speed = speed

        # RNN loss - now handled by call to SimpleRNN
        #rnn_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.dt)
        #self.add_loss('RNN', loss_weight=0.1, loss=gru_loss)
        self.add_loss('targetVelocity', loss_weight=1, loss=VelocityLoss())

        motion_onset_range = np.array(motion_onset_range) / self.dt
        self.motion_onset_range = [int(motion_onset_range[0]), int(motion_onset_range[1])]
        self.delay_range = self.motion_onset_range

    def generate(self, batch_size, n_timesteps):
        angle_set = np.deg2rad(np.arange(0, 360, self.angular_step))
        reps = int(np.ceil(batch_size / len(angle_set)))
        angle = np.tile(angle_set, reps=reps)
        batch_size = reps * len(angle_set)

        target_vectors = self.speed * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        print(n_timesteps)
        target_vectors_time = np.repeat(target_vectors[:,np.newaxis,:], n_timesteps, axis=1)
        for i in range(batch_size):
            motion_time = int(np.random.uniform(self.motion_onset_range[0], self.motion_onset_range[1]))
            target_vectors_time[i, :motion_time, :] = np.zeros(target_vectors_time[i, :motion_time, :].shape)
            targets = target_vectors_time

        print(targets.shape)
        return [
            {"inputs": targets},
            self.convert_to_tensor(targets)
        ]

class SmoothPursuitData(Task):
    """
    """

    def __init__(
            self,
            dt: float = 0.01,
            name: str = 'SmoothPursuitData',
            angular_step: float = 45,
            catch_trial_perc: float = 50,
            speed: float = 0.1,
            start_position: Union[list, tuple, np.ndarray] = None,
            deriv_weight: float = 0.,
            motion_onset_range: Union[list, tuple, np.ndarray] = (0.05, 0.25),
            **kwargs
    ):

        super().__init__(dt=dt, name=name, **kwargs)

        self.angular_step = angular_step
        self.catch_trial_perc = catch_trial_perc
        self.speed = speed

        # RNN loss - now handled by call to SimpleRNN
        #rnn_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.dt)
        #self.add_loss('RNN', loss_weight=0.1, loss=gru_loss)
        self.add_loss('targetVelocity', loss_weight=1, loss=VelocityLoss())

        motion_onset_range = np.array(motion_onset_range) / self.dt
        self.motion_onset_range = [int(motion_onset_range[0]), int(motion_onset_range[1])]
        self.delay_range = self.motion_onset_range

    def generate(self, batch_size, n_timesteps):
        f = open('/home/seth/Temp/test.bin', "rb")
        data = np.fromfile(f, dtype=np.float64, sep='')
        shape = data[-3:].astype(int)
        data2 = data[:-3]
        data3 = data2.reshape(shape)
        number_of_rows = data3.shape[0]
        random_indices = np.random.choice(number_of_rows,
                                          size=batch_size,
                                          replace=True)

        targets = data3[random_indices,:n_timesteps,0:3]
        outputs = data3[random_indices,:n_timesteps,3:]
        print(targets.shape)
        print(outputs.shape)

        return [
            {"inputs": targets},
            self.convert_to_tensor(outputs)
        ]
