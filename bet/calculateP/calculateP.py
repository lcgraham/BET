# Copyright (C) 2014-2015 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_emulated` provides a skeleton class and calculates
    the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob_samples_mc` estimates the volumes of
    the voronoi cells using MC integration
"""
import scipy.stats
import numpy as np
import scipy.spatial as spatial
import bet.util as util
from bet.Comm import comm, MPI 

def emulate_iid_normal(num_l_emulate, mean, covariance):
    """
    Sample the parameter space using emulated samples drawn from a multivariate
    normal distribution. These samples are iid so that we can apply the
    standard MC assumuption/approximation. See
    :meth:`numpy.random.multivariate_normal`.

    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int
    :param mean: The mean of the n-dimensional distribution.
    :type mean: :class:`numpy.ndarray` of shape (ndim, )
    :param covariance: The covariance of the n-dimensional distribution.
    :type covariance: 2-D :class:`numpy.ndarray` of shape (ndim, ndim)

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    """
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    mean = util.fix_dimensions_vector(mean)
    if not isinstance(covariance, np.ndarray):
        covariance = np.array([[covariance]])
    lambda_emulate = np.random.multivariate_normal(mean, covariance,
            num_l_emulate)
    return lambda_emulate 

def emulate_iid_truncnorm(num_l_emulate, mean, covariance, input_domain):
    """
    Sample the parameter space using emulated samples drawn from a 
    truncated normal distribution. These samples are iid so that we can apply
    the
    standard MC assumuption/approximation. See
    :meth:`scipy.stats.truncnorm`.

    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int
    :param mean: The mean of the n-dimensional distribution.
    :type mean: :class:`numpy.ndarray` of shape (ndim, )
    :param covariance: The covariance of the n-dimensional distribution.
    :type covariance: 2-D :class:`numpy.ndarray` of shape (ndim, ndim)
    :param input_domain: The domain for each parameter of the model.
    :type input_domain: :class:`~numpy.ndarray` of shape (dim, 2). Note that
        ``dim==1``.

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    
    """
    # TODO update using tensor product to create a multivarite normal
    a = (input_domain[:, 0] - mean) / np.sqrt(covariance)
    b = (input_domain[:, 1] - mean) / np.sqrt(covariance)
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    lambda_emulate = scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=np.sqrt(covariance),
            size=num_l_emulate)
    return np.expand_dims(lambda_emulate, 1)

class dim_not_matching(Exception):
    """
    Exception for when the dimension is inconsistent.
    """

def emulate_iid_lebesgue(lam_domain, num_l_emulate):
    """
    Sample the parameter space using emulated samples drawn from a uniform
    Lesbegue measure. These samples are iid so that we can apply the standard
    MC assumuption/approximation. See :meth:`numpy.random.random`.

    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)  
    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    """
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    lam_width = lam_domain[:, 1] - lam_domain[:, 0]
    lambda_emulate = lam_width*np.random.random((num_l_emulate,
        lam_domain.shape[0]))+lam_domain[:, 0] 
    return lambda_emulate

def emulate_iid_beta(a, b, lam_domain, num_l_emulate):
    """
    Sample the parameter space using emulated samples drawn from a beta
    distribution. These samples are iid so that we can apply the standard
    MC assumuption/approximation. See :meth:`numpy.random.beta`.

    .. note::
        
        See :class:`scipy.stats.dirichlet` for the multivariate extension of
        the Beta function. This has not yet need implemented.

    :param a float: alpha
    :param b float: beta
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)  
    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    """
    # TODO implement multivariate version
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    lam_width = lam_domain[:, 1] - lam_domain[:, 0]
    lambda_emulate = lam_width*np.random.beta(a, b, (num_l_emulate,
        lam_domain.shape[0]))+lam_domain[:, 0] 
    return lambda_emulate 

def prob_emulated(samples, data, rho_D_M, d_distr_samples,
        lambda_emulate=None, d_Tree=None): 
    r"""

    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}})`, the
    probability assoicated with a set of voronoi cells defined by
    ``num_l_emulate`` iid samples :math:`(\lambda_{emulate})`.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :rtype: tuple
    :returns: (P, lambda_emulate, io_ptr, emulate_ptr, lam_vol)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)
    
    # Calculate Probabilties
    P = np.zeros((lambda_emulate.shape[0],))
    d_distr_emu_ptr = np.zeros(emulate_ptr.shape)
    d_distr_emu_ptr = io_ptr[emulate_ptr]
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(d_distr_emu_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P[Itemp] = rho_D_M[i]/Itemp_sum

    return (P, lambda_emulate, io_ptr, emulate_ptr)

def prob(samples, data, rho_D_M, d_distr_samples, d_Tree=None): 
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are assumed to be equal under the MC assumption.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, io_ptr) where P is the
        probability associated with samples, and lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins.

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)

    # Set up local arrays for parallelism
    local_index = np.array_split(np.arange(samples.shape[0]),
            comm.size)[comm.rank]
    samples_local = samples[local_index, :]
    data_local = data[local_index, :]
    local_array = np.array(local_index, dtype='int64')
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data_local)

    # Apply the standard MC approximation and
    # calculate probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]

    lam_vol = (1.0/float(samples.shape[0]))*np.ones((samples.shape[0],))

    return (P, lam_vol, io_ptr)

def prob_mc(samples, data, rho_D_M, d_distr_samples,
            lambda_emulate=None, d_Tree=None): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are approximated using MC integration.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to estimate the volumes of the Voronoi
        cells associated with ``samples``

    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if d_Tree is None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)

    lam_vol, lam_vol_local, local_index = estimate_volume(samples,
            lambda_emulate)

    local_array = np.array(local_index, dtype='int64')
    data_local = data[local_index, :]
    samples_local = samples[local_index, :]
    
    
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr_local) = d_Tree.query(data_local)

    # Calculate Probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr_local, i)
        Itemp_sum = np.sum(lam_vol_local[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]*lam_vol_local[Itemp]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]
    return (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr)

def estimate_volume(samples, lambda_emulate=None, p=2):
    r"""
    Estimate the volume fraction of the Voronoi cells associated with
    ``samples`` using ``lambda_emulate`` as samples for Monte Carlo
    integration. Specifically we are estimating 
    :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
    
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :param int p: p for the L-p norm 
    
    :rtype: tuple
    :returns: (lam_vol, lam_vol_local, local_index) where ``lam_vol`` is the
        global array of volume fractions, ``lam_vol_local`` is the local array
        of volume fractions, and ``local_index`` a list of the global indices
        for local arrays on this particular processor ``lam_vol_local =
        lam_vol[local_index]``
    
    """

    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if lambda_emulate is None:
        lambda_emulate = samples
 
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate, p=p)

    # Apply the standard MC approximation to determine the number of emulated
    # samples per model run sample. This is for approximating 
    # \mu_Lambda(A_i \intersect b_j)
    lam_vol = np.zeros((samples.shape[0],)) 
    for i in range(samples.shape[0]):
        lam_vol[i] = np.sum(np.equal(emulate_ptr, i))
    clam_vol = np.copy(lam_vol) 
    comm.Allreduce([lam_vol, MPI.DOUBLE], [clam_vol, MPI.DOUBLE], op=MPI.SUM)
    lam_vol = clam_vol
    num_emulated = lambda_emulate.shape[0]
    num_emulated = comm.allreduce(num_emulated, op=MPI.SUM)
    lam_vol = lam_vol/(num_emulated)

    # Set up local arrays for parallelism
    local_index = np.array_split(np.arange(samples.shape[0]),
            comm.size)[comm.rank]
    local_index = np.array(local_index, dtype='int64')
    lam_vol_local = np.array_split(lam_vol, comm.size)[comm.rank]

    return (lam_vol, lam_vol_local, local_index)

def estimate_local_volume(samples, input_domain=None, num_l_emulate_local=1e2,
        p=2, max_num_l_emulate=1e3, distribution='uniform',
        a=None, b=None):
    r"""

    Exactly calculates the volume fraction of the Voronoice cells associated
    with ``samples``. Specifically we are calculating 
    :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.

    Volume of the L-p ball is obtained from  Wang, X.. (2005). Volumes of
    Generalized Unit Balls. Mathematics Magazine, 78(5), 390–395.
    `DOI 10.2307/30044198 <http://doi.org/10.2307/30044198>`_

    .. todo::
        
        Implement beta, normal, truncated normal distributions
    
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param input_domain: The limits of the domain :math:`\mathcal{D}`.
    :type input_domain: :class:`numpy.ndarray` of shape (ndim, 2)
    :param int num_l_emulate: The number of emulated samples.
    :param int p: p for the L-p norm 
    :param string distribution: Probability distribution (uniform, normal,
        truncnorm, beta)
    :param float a: mean or alpha (normal/truncnorm, beta)
    :param float b: covariance or beta (normal/truncnorm, beta)

    :rtype: tuple
    :returns: (lam_vol, lam_vol_local, local_index) where ``lam_vol`` is the
        global array of volume fractions, ``lam_vol_local`` is the local array
        of volume fractions, and ``local_index`` a list of the global indices
        for local arrays on this particular processor ``lam_vol_local =
        lam_vol[local_index]``
    
    """
    # TODO this might work better if we first normalize the domain
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 

    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)

    # for each sample
    # determine the appropriate radius of the Lp ball (this should be the
    # distance to the farthest neighboring Voronoi cell)

    # calculating this exactly is hard so we will estimate it as follows
    # TODO it is unclear whether to use min, mean, or the first n nearest
    # samples
    # Calculate the pairwise distances
    pairwise_distance = spatial.distance.pdist(samples, p=p)
    pairwise_distance = spatial.distance.squareform(pairwise_distance)
    pairwise_distance_ma = np.ma.masked_less_equal(pairwise_distance, 0.)
    # Calculate mean, std of pairwise distances
    sample_radii = np.std(pairwise_distance_ma, 0)*3

    # determine the volume of the Lp ball
    dim = samples.shape[1]
    sample_Lp_ball_vol = sample_radii**dim * scipy.special.gamma(1+1./p) / \
            scipy.special.gamma(1+float(dim)/p)


    lam_vol = np.zeros((samples.shape[0],)) 

    # Set up local arrays for parallelism
    local_index = np.array_split(np.arange(samples.shape[0]),
            comm.size)[comm.rank]
    local_index = np.array(local_index, dtype='int64')

    # parallize 
    for i, iglobal in enumerate(local_index):
        samples_in_cell = 0
        total_samples = 1e2
        while samples_in_cell < num_l_emulate_local:
            total_samples = total_samples*10
            # Sample within an Lp ball until num_l_emulate_local samples are present in
            # the ball
            # TODO implement this for real
            local_samples = np.random.random((total_samples, dim))

            # determine the number of samples in the Voronoi cell (intersected
            # with the input_domain)
            if input_domain is not None:
                left = np.repeat(input_domain[:, 0], total_samples, 0)
                right = np.repeat(input_domain[:, 1], total_samples, 0)
                left = np.all(np.greater_equal(local_samples, left), axis=1)
                right = np.all(np.greater_equal(local_samples, right), axis=1)
                inside = np.logical_and(left, right)
                lambda_emulate = lambda_emulate[inside, :]

            (_, emulate_ptr) = l_Tree.query(lambda_emulate, p=p,
                    distance_upper_bound=sample_radii[iglobal])
            samples_in_cell = np.sum(np.equal(emulate_ptr, iglobal))

        # the volume for the Voronoi cell corresponding to this sample is the
        # the volume of the Lp ball times the ratio
        # "num_samples_in_cell/num_total_local_emulated_samples" 
        lam_vol_local[i] = sample_Lp_ball_vol[iglobal]*float(samples_in_cell)/total_samples
    
    lam_vol = util.get_global_values(lam_vol_local)

    return (lam_vol, lam_vol_local, local_index)


def exact_volume_1D(samples, input_domain, distribution='uniform', a=None,
        b=None): 
    r"""

    Exactly calculates the volume fraction of the Voronoice cells associated
    with ``samples``. Specifically we are calculating 
    :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
    
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param input_domain: The limits of the domain :math:`\mathcal{D}`.
    :type input_domain: :class:`numpy.ndarray` of shape (ndim, 2)
    :param string distribution: Probability distribution (uniform, normal,
        truncnorm, beta)
    :param float a: mean or alpha (normal/truncnorm, beta)
    :param float b: covariance or beta (normal/truncnorm, beta)

    :rtype: tuple
    :returns: (lam_vol, lam_vol_local, local_index) where ``lam_vol`` is the
        global array of volume fractions, ``lam_vol_local`` is the local array
        of volume fractions, and ``local_index`` a list of the global indices
        for local arrays on this particular processor ``lam_vol_local =
        lam_vol[local_index]``
    
    """

    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)

    #if sample_obj.get_dim() != 1:
    if samples.shape[1] != 1:
        raise dim_not_matching("Only applicable for 1D domains.")

    # sort the samples
    sort_ind = np.squeeze(np.argsort(samples, 0))
    sorted_samples = samples[sort_ind]
    domain_width = input_domain[:, 1] - input_domain[:, 0]

    # determine the mid_points which are the edges of the associated voronoi
    # cells and bound the cells by the domain
    edges = np.concatenate(([input_domain[:, 0]], (sorted_samples[:-1, :] +\
        sorted_samples[1:, :])*.5, [input_domain[:, 1]]))
    if distribution == 'normal':
        edges = scipy.stats.norm.cdf(edges, loc=a, scale=np.sqrt(b))
    elif distribution == 'truncnorm':
        l = (input_domain[:, 0] - a) / np.sqrt(b)
        r = (input_domain[:, 1] - a) / np.sqrt(b)
        edges = scipy.stats.truncnorm.cdf(edges, a=l, b=r, loc=a, scale=np.sqrt(b))
    elif distribution == 'beta':
        edges = scipy.stats.beta.cdf(edges, a=a, b=b,
                loc=input_domain[:, 0], scale=domain_width)
    # calculate difference between right and left of each cell and renormalize
    sorted_lam_vol = np.squeeze(edges[1:, :] - edges[:-1, :])
    lam_vol = np.zeros(sorted_lam_vol.shape)
    lam_vol[sort_ind] = sorted_lam_vol
    if distribution == 'uniform':
        lam_vol = lam_vol/domain_width
    # Set up local arrays for parallelism
    local_index = np.array_split(np.arange(samples.shape[0]),
            comm.size)[comm.rank]
    local_index = np.array(local_index, dtype='int64')
    lam_vol_local = np.array_split(lam_vol, comm.size)[comm.rank]
    return (lam_vol, lam_vol_local, local_index)
    

    
    
