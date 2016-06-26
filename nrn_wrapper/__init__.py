from __future__ import division
from neuron import nrn, h
import json
import numpy as np
import sys
from utilities import isfloat

__author__ = 'caro'


class Section(nrn.Section):
    """
    :cvar PROXIMAL: Number corresponding to the origin of a Section.
    :type PROXIMAL: float
    :cvar DISTAL: Number corresponding to the end of a Section.
    :type DISTAL: float

    :Examples:
    soma = Section('soma', None, geom={'L': 30, 'diam': 30}, mechanisms={'pas': {'0.5': {}}})
    """

    PROXIMAL = 0
    DISTAL = 1

    def __init__(self, name, index, geom, nseg=1, Ra=100, cm=1, mechanisms=None, parent=None,
                 connection_point=DISTAL):
        """
        Initialization of a Section. Inherits from nrn.Section.

        :param name: Name of the section (either soma, dendrites or axon_secs).
        :type name: str
        :param index: Index of the section in the list of dendrites or axon_secs.
        :type index: int
        :param geom: Geometry of the Section. Defined either by length (um) and diameter (um) in a dictionary or by a
        list of [x, y, z, diam] coordinates.
        :type geom: dict or list
        :param nseg: Number of segments. Specifies the number of internal points at which NEURON computes solutions.
        :type nseg: int
        :param Ra: Axial resistance (Ohm * cm).
        :type Ra: float
        :param cm: Membrane capacitance (uF * cm^2).
        :type cm: float
        :param mechanisms: Mechanisms (e.g. ion channels). Nested dictionary arranged first by mechanism name,
        second by pos (position on the section) having the mechanism parameters as values. Reversal potentials are set
        as 'ena', 'ek', etc. in mechanisms of sections with the name 'na_ion', 'k_ion', etc.
        :type mechanisms: dict
        :param parent: Section to which this Section is connected.
        :type parent: Section
        :param connection_point: Point of the connection on the parent section (between 0 and 1).
        :type connection_point: float
        :return: Section.
        :rtype: Section
        """

        nrn.Section.__init__(self)  # important for inheritance from NEURON

        # name and index (for later retrieval)
        self.name = name
        self.index = index

        # set geometry
        if 'L' in geom and 'diam' in geom:
            self.L = geom['L']
            self.diam = geom['diam']
        else:
            self.set_geom(geom)
        self.nseg = nseg

        # set cable properties
        self.Ra = Ra
        self.cm = cm

        # connect to parent section
        self.connection_point = None
        if parent:
            self.connect(parent, connection_point)

        # add ion channels
        if mechanisms is not None:
            for mechanism_name, mechanism_params in mechanisms.iteritems():
                self.insert(mechanism_name)
                for pos, params in mechanism_params.iteritems():
                    if params is not None:
                        mechanism = getattr(self(float(pos)), mechanism_name)
                        for name, v in params.iteritems():
                            if '_ion' in mechanism_name or 'pas' in mechanism_name or mechanism_name in name:
                                setattr(self(float(pos)), name, v)
                            else:
                                setattr(mechanism, name, v)

        # add spike_count
        self.spike_count = None  # sustains NEURON reference for recording spikes (see :func Cell.record_spikes)

    def set_geom(self, geom):
        """
        Create 3d geometry of the neuron.

        :param geom: List of 4d coordinates [x, y, z, diam].
        :type geom: list
        """
        self.push()  # necessary to access Section in NEURON
        h.pt3dclear()
        for g in geom:
            h.pt3dadd(g[0], g[1], g[2], g[3])
        h.pop_section()  # restore the previously accessed Section

    def connect(self, parent, connection_point):
        """
        Connect the proximal end of this section to the connection point at the parent section.
        :param parent: Section to connect to.
        :type parent: Section
        :param connection_point: Point at which to attach this section on the parent.
        :type connection_point: float
        """
        super(Section, self).connect(parent, connection_point, self.PROXIMAL)
        self.connection_point = connection_point
        
    def get_dict(self, geom_type):
        """
        Create a dictionary representation of this section.
        :param geom_type: 'stylized': dict with L and diam, '3d': list of [x, y, z, diam]
        :type geom_type: str
        :return: Dictionary containing all parameters of a section.
        :rtype: dict
        """
        
        # geometry
        if geom_type == 'stylized':
            geom = {'L': self.L, 'diam': self.diam}
        elif geom_type == '3d':
            geom = [[h.x3d(seg.x, sec=self), h.y3d(seg.x, sec=self), h.z3d(seg.x, sec=self), seg.diam]
                    for seg in self]
        else:
            raise ValueError("Invalid geometry type! Should be 'stylized' or '3d'.")

        # mechanisms
        mechanism_names = [mechanism_name for mechanism_name in vars(self(.5))
                           if isinstance(getattr(self(.5), mechanism_name), nrn.Mechanism)]
        mechanisms = {name: dict() for name in mechanism_names}  # saved according to name, then pos
        for mechanism_name in mechanism_names:
            for seg in self:
                mechanisms[mechanism_name][str(seg.x)] = \
                    {p: getattr(seg, p) for p in vars(getattr(seg, mechanism_name))}
        
        # parent
        if h.SectionRef(sec=self).has_parent():
            parent = [h.SectionRef(sec=self).parent.name, h.SectionRef(sec=self).parent.index]
        else:
            parent = None

        return {'nseg': self.nseg, 'Ra': self.Ra, 'cm': self.cm,
                             'geom': geom, 'mechanisms': mechanisms, 'parent': parent,
                             'connection_point': self.connection_point}

    def record(self, var, pos=0.5):
        """
        Records any variable var.
        Note: Values are updated after each NEURON h.run().

        :param: var: Name of the variable to record.
        :type: var: str
        :param: pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type: pos: float
        :return: vec: Vector containing the values of the recorded variable after h.run().
        :rtype: vec: h.Vector
        """
        vec = h.Vector()
        vec.record(getattr(self(pos), '_ref_'+var))
        return vec

    def record_from(self, mechanism, var, pos=.5):
        """
        Record any variable of a mechanism.
        Note: Values are updated after each NEURON h.run().

        :param mechanism: Name of the mechanism.
        :type mechanism: str
        :param var: Name of the variable.
        :type var: str
        :param pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type pos: float
        :return: Vector containing the values of the recorded variable after h.run().
        :rtype: h.Vector
        """
        vec = h.Vector()
        vec.record(getattr(getattr(self(pos), mechanism), '_ref_'+var))
        return vec

    def record_spikes(self, threshold=-30, pos=.5):
        """
        Records the spikes of this section.
        Note: Values are updated after each NEURON h.run().

        :param: pos: Indicates the position on the Section at which is recorded (number between 0 and 1).
        :type: pos: float
        :param: threshold: Only spikes above this threshold are counted as spikes.
        :type: threshold: float
        :return: vec: Contains the times where spikes occurred
        :rtype: vec: h.Vector
        """
        vec = h.Vector()
        self.spike_count = h.APCount(pos, sec=self)  # spike_count assigned to self to keep NEURON reference
        self.spike_count.thresh = threshold
        self.spike_count.record(vec)
        return vec

    def inject_stepcurrent(self, i_amp, delay, dur, pos=.5):
        """
        Inject a step current of specified amplitude, delay and duration.
        Note: Keep all output variables for NEURON to reference.

        :param i_amp: Amplitude of the step current in nA.
        :type i_amp: float
        :param delay: Time before the step.
        :type delay: float
        :param dur: Duration of the step.
        :type dur: float
        :param pos: Indicates the position of the IClamp on the Section (number between 0 and 1).
        :type pos: float
        :return: IClamp (NEURON needs the reference).
        :rtype: h.IClamp
        """
        stim = h.IClamp(pos, sec=self)
        stim.amp = i_amp
        stim.delay = delay
        stim.dur = dur
        return stim

    def play_current(self, i_amp, t, pos=.5):
        """
        At each time step inject a current equivalent to i_amp at this time step.
        Note: Keep all output variables for NEURON to reference.

        :param: pos: Indicates the position of the IClamp on the Section (number between 0 and 1).
        :type: pos: float
        :param i_amp: Current injected at each time step in nA.
        :type i_amp: array_like
        :param t: Time at each time step.
        :type t: array_like
        :return: IClamp and the current and time vector (NEURON needs the reference).
        :rtype: h.IClamp, h.Vector, h.Vector
        """

        stim = h.IClamp(pos, sec=self)
        stim.delay = 0  # 0 necessary for playing the current into IClamp
        stim.dur = 1e9  # 1e9 necessary for playing the current into IClamp
        i_vec = h.Vector()
        i_vec.from_python(i_amp)
        t_vec = h.Vector()
        t_vec.from_python(np.concatenate((np.array([0]), t)))  # at time 0 no current is played
        i_vec.play(stim._ref_amp, t_vec)  # play current into IClamp (use experimental current trace)
        return stim, i_vec, t_vec


class Cell(object):

    def __init__(self, model, mechanism_dir=None):
        """
        Initialization of a Cell.
        Note: The cell is divided into soma (a Section), dendrites and axon_secs (both list of Sections). Each Section
        can be defined in the model specification as dictionary of the input parameters of a Section.

        :param model: Specification of a Cell (see note and examples).
        :type model: dict
        :param mechanism_dir: Path to the .mod files of the mechanisms. (Can be defined irrespective of the machine by
        complete_mechanismdir).
        :type mechanism_dir: str
        :return: Cell as defined in model.
        :rtype: Cell

        :Examples:
        cell = Cell({"soma": {
                        "mechanisms": {
                            "pas": {"0.5": {}}
                          },
                        "geom": {
                            "diam": 15,
                            "L": 10
                        },
                        "Ra": 100.0,
                        "cm": 1.0,
                        "nseg": 1
                    }})
        """

        # load mechanisms (ion channel implementations)
        if mechanism_dir is not None:
            h.nrn_load_dll(mechanism_dir)  # must be loaded before insertion of Mechanisms! (cannot be loaded twice)

        # create Cell with given parameters
        self.__create(model)

    @classmethod
    def from_modeldir(cls, model_dir, mechanism_dir=None):
        """
        Alternative method for initialization from a modeldir.
        :param model_dir: Path to the cell model.
        :type model_dir: str
        :param mechanism_dir: Path to the .mod files of the mechanisms. (Can be defined irrespective of the machine by
        complete_mechanismdir).
        :type mechanism_dir: str
        :return: Cell as defined in the file at modeldir.
        :rtype: Cell
        """
        with open(model_dir, 'r') as f:
            params = json.load(f)

        return cls(params, mechanism_dir)

    def __create(self, model):
        """
        Creates a cell from the model specification.

        :param model: Model specification.
        :type model: dict
        """

        # set temperature
        if 'celsius' in model:
            self.celsius = model['celsius']
            h.celsius = self.celsius  # set celsius also in hoc

        # create sections
        self.soma = Section('soma', None, **model['soma'])

        if 'dendrites' in model:
            self.dendrites = [0] * len(model['dendrites'])
            for i in range(len(model['dendrites'])):
                params_sec = model['dendrites'][str(i)]
                if 'parent' in model['dendrites'][str(i)].keys():
                    params_sec['parent'] = self.substitute_section(params_sec['parent'][0], params_sec['parent'][1])
                self.dendrites[i] = Section('dendrites', i, **params_sec)

        if 'axon_secs' in model:
            self.axon_secs = [0] * len(model['axon_secs'])
            for i in range(len(model['axon_secs'])):
                params_sec = model['axon_secs'][str(i)]
                if 'parent' in model['axon_secs'][str(i)].keys():
                    params_sec['parent'] = self.substitute_section(params_sec['parent'][0], params_sec['parent'][1])
                self.axon_secs[i] = Section('axon_secs', i, **params_sec)

    @staticmethod
    def __get_attr_tmp(var, key):
        if key.isdigit():  # should be the index of the section
            return var[int(key)]
        elif isfloat(key):  # should be pos applied to section
            return var(float(key))
        else:
            return getattr(var, key)

    def update_attr(self, keys, value):
        """
        Updates the value of an attribute defined by a list of keys.
        Note: Attribute needs to exist already.

        :param keys: List of keys leading to the attribute in self.params.
        :type keys: list of str
        :param value: New value of the attribute.
        :type value: type depends on the attribute
        """
        setattr(reduce(self.__get_attr_tmp, [self] + keys[:-1]), keys[-1], value)

    def get_attr(self, keys):
        """
        Returns the value of the attribute indexed by keys.

        :param keys: List of keys leading to the attribute in self.params.
        :type keys: list of str
        :return: Value of the accessed attribute.
        :rtype: type depends on the attribute
        """
        return reduce(self.__get_attr_tmp, [self] + keys)

    def substitute_section(self, name, index):
        """
        Substitutes the name and index of a 'parent' section with the corresponding Section of the Cell.

        :param params: Part of the Cell params containing the parent name and index.
        :type params: dict
        :return: New parameters with substituted Section
        :rtype: dict
        """
        if index is None or np.isnan(index):
            return getattr(self, name)
        else:
            return getattr(self, name)[index]

    def get_dict(self, geom_type='stylized'):
        cell_dict = dict()

        # get sections
        cell_dict['soma'] = self.soma.get_dict(geom_type)

        if hasattr(self, 'dendrites'):
            cell_dict['dendrites'] = dict()
            for i, dendrite in enumerate(cell.dendrites):
                cell_dict['dendrites'][str(i)] = self.dendrites[i].get_dict(geom_type)
        if hasattr(self, 'axon_secs'):
            cell_dict['axon_secs'] = dict()
            for i, dendrite in enumerate(cell.dendrites):
                cell_dict['axon_secs'][str(i)] = self.dendrites[i].get_dict(geom_type)

        return cell_dict


def complete_mechanismdir(mechanism_dir):
    if sys.maxsize > 2**32:
        mechanism_dir += '/x86_64/.libs/libnrnmech.so'
    else:
        mechanism_dir += '/i686/.libs/libnrnmech.so'
    return mechanism_dir


if __name__ == "__main__":

    # create Section
    soma = Section('soma', None, geom={'L':30, 'diam':30}, mechanisms={'pas': {'0.5': {}}})

    # create Cell
    cell = Cell({"soma": {
                        "mechanisms": {
                            "pas": {"0.5": {}}
                          },
                        "geom": {
                            "diam": 15,
                            "L": 10
                        },
                        "Ra": 100.0,
                        "cm": 1.0,
                        "nseg": 1
                    }})

    # update and get attribute
    cell.update_attr(['dendrites', '0', '0.5', 'pas', 'g'], 0.001)

    print cell.get_attr(['dendrites', '0', '0.5', 'pas', 'g'])

    # load cell from modeldir
    cell = Cell.from_modeldir('./cells/test_cell.json', complete_mechanismdir('./fitted_channels/'))

    # save cell
    with open('./cells/test_cell.json', 'w') as f:
        json.dump(cell.get_dict(), f, indent=4)

    # reload saved cell
    cell = Cell.from_modeldir('./test_cell_new.json')


    # TODO: celsius not part of cell, define elsewhere! belongs more to experimental protocol!