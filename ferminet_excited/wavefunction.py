from ferminet import networks
from ferminet import envelopes
from ferminet import psiformer
import importlib

def create_network(cfg, charges, nspins):
    if cfg.network.make_feature_layer_fn:
        feature_layer_module, feature_layer_fn = (
            cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
        feature_layer_module = importlib.import_module(feature_layer_module)
        make_feature_layer: networks.MakeFeatureLayer = getattr(
            feature_layer_module, feature_layer_fn
        )
        feature_layer = make_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.rescale_inputs,
        )

    if cfg.network.make_envelope_fn:
        envelope_module, envelope_fn = (
            cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
        envelope_module = importlib.import_module(envelope_module)
        make_envelope = getattr(envelope_module, envelope_fn)
        envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
    else:
        envelope = envelopes.make_isotropic_envelope()

    if cfg.network.network_type == 'ferminet':
        network = networks.make_fermi_net(
            nspins,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.jastrow,
            bias_orbitals=cfg.network.bias_orbitals,
            full_det=cfg.network.full_det,
            rescale_inputs=cfg.network.rescale_inputs,
            complex_output=cfg.network.complex,
            hidden_dims=cfg.network.ferminet.hidden_dims,
            use_last_layer=cfg.network.ferminet.use_last_layer,
            separate_spin_channels=cfg.network.ferminet.separate_spin_channels,
            schnet_electron_electron_convolutions=cfg.network.ferminet.schnet_electron_electron_convolutions,
            electron_nuclear_aux_dims=cfg.network.ferminet.electron_nuclear_aux_dims,
            schnet_electron_nuclear_convolutions=cfg.network.ferminet.schnet_electron_nuclear_convolutions,
            # **cfg.network.ferminet,
        )
    elif cfg.network.network_type == 'psiformer':
        network = psiformer.make_fermi_net(
            nspins,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=cfg.network.get('complex', False),
            **cfg.network.psiformer,
        )
    
    return network