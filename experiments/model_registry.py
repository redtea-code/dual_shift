"""训练入口共享：backbone + comparison + causal 模型注册与 config 注入。"""
from Model import resnet18_film, resnet18_daft, resnet10_daft
from Model.backbone.film_backbone import resnet18_e1, resnet10_film, resnet10_ce_only, resnet_light_film
from Model import (
    vit_tiny_film, vit_small_film,
    vit_tiny_daft, vit_small_daft,
    vit_tiny_backdoor, vit_small_backdoor,
    resnet10_backdoor, resnet18_backdoor, resnet34_backdoor,
    resnet10_disentangled, resnet18_disentangled,
)
from Model.comparison.factories import MODEL_REGISTRY as COMPARISON_MODELS



BACKBONE_MODELS = {
    'resnet18_film': (resnet18_film, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=None, film_stages='last', get_feature=False)),
    'resnet10_film': (resnet10_film, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=None, film_stages='last', feature=False)),
    'resnet10_ce_only': (resnet10_ce_only, dict(
        txt_dim=0, num_classes=3, pretrained_weights=None, feature=False,
    )),
    'resnet18_daft': (resnet18_daft, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=None, feature=False)),
    'resnet10_daft': (resnet10_daft, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=None, feature=False)),
    'resnet18_backdoor': (resnet18_backdoor, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=None)),
    'resnet10_backdoor': (resnet10_backdoor, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=None)),
    'resnet34_backdoor': (resnet34_backdoor, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=None)),
    'vit_tiny_backdoor': (vit_tiny_backdoor, dict(txt_dim=0, num_classes=3,
                                                  pretrained_weights=False, get_feature=False)),
    'vit_small_backdoor': (vit_small_backdoor, dict(txt_dim=0, num_classes=3,
                                                    pretrained_weights=False, get_feature=False)),
    'vit_tiny_film': (vit_tiny_film, dict(txt_dim=0, num_classes=3,
                                          pretrained_weights=False, get_feature=False)),
    'vit_small_film': (vit_small_film, dict(txt_dim=0, num_classes=3,
                                            pretrained_weights=False, get_feature=False)),
}

CAUSAL_MODELS = {
    'resnet18_disentangled': (resnet18_disentangled, dict(
        txt_dim=0, num_classes=3, sub_dim=256, z_dim=128,
        pretrained_weights=None, causal_phase=1,
    )),
    'resnet10_disentangled': (resnet10_disentangled, dict(
        txt_dim=0, num_classes=3, sub_dim=256, z_dim=128,
        pretrained_weights=None, causal_phase=1,
    )),
}

# C2: core registry lives in Model layer; experiments adds BACKBONE fallback.
from Model.causal.confound_registry import (
    AGE_CONFOUND_MODELS,
    build_age_confound_model as _build_age_confound_model_core,
    print_age_confound_model_list,
)

ALL_MODELS = set(BACKBONE_MODELS) | set(COMPARISON_MODELS) | set(CAUSAL_MODELS)


def _inject_backbone_config(model_name, kwargs, cf):
    mk = dict(kwargs)
    mk['txt_dim'] = cf.get('table_feature', 0)
    img_sz = tuple(cf.get('img_sz', (160, 160, 96)))
    # Only models that accept these keys should receive them.
    if 'backdoor' in model_name or 'disentangled' in model_name:
        mk['descripe'] = cf.get('descripe', 'none')
    if 'backdoor' in model_name:
        mk['img_size'] = img_sz
        mk['use_class_head'] = cf.get('use_class_head', False)
        mk['class_head_kwargs'] = cf.get('class_head_kwargs', None)
        if 'backdoor_kwargs' in cf:
            mk['backdoor_kwargs'] = cf['backdoor_kwargs']
    elif 'film' in model_name:
        # Factory uses `feature`, not get_feature; default factory is feature=True.
        mk['feature'] = False
        if 'film_stages' in cf:
            mk['film_stages'] = cf['film_stages']
    elif 'daft' in model_name:
        mk['feature'] = False
    elif 'disentangled' in model_name:
        mk['img_size'] = img_sz
        causal_cfg = cf.get('causal') or {}
        if 'sub_dim' in causal_cfg:
            mk['sub_dim'] = causal_cfg['sub_dim']
        if 'z_dim' in causal_cfg:
            mk['z_dim'] = causal_cfg['z_dim']
        if causal_cfg.get('use_age_prediction', False):
            mk['use_age_prediction'] = True
        if causal_cfg.get('use_age_adversarial', False):
            mk['use_age_adversarial'] = True
        if 'age_head_hidden' in causal_cfg:
            mk['age_head_hidden'] = causal_cfg['age_head_hidden']
        if 'age_adv_hidden' in causal_cfg:
            mk['age_adv_hidden'] = causal_cfg['age_adv_hidden']
        if 'grl_lambda' in causal_cfg:
            mk['grl_lambda'] = causal_cfg['grl_lambda']
        gamma_mech = causal_cfg.get('gamma_mech') or {}
        if gamma_mech:
            mk['gamma_mech_mode'] = gamma_mech.get('mode', 'learned')
            if 'constant_value' in gamma_mech:
                mk['gamma_constant_value'] = gamma_mech['constant_value']
            if 'shuffle_seed' in gamma_mech:
                mk['gamma_shuffle_seed'] = gamma_mech['shuffle_seed']
        if 'fusion_mode' in causal_cfg:
            mk['fusion_mode'] = causal_cfg['fusion_mode']
        lw = causal_cfg.get('loss_weights') or {}
        if float(lw.get('age_prediction', 0) or 0) > 0:
            mk['use_age_prediction'] = True
        if float(lw.get('age_adversarial', 0) or 0) > 0:
            mk['use_age_adversarial'] = True
    elif 'vit_' in model_name:
        mk['img_size'] = img_sz
    return mk


def _inject_comparison_config(model_name, kwargs, cf):
    mk = dict(kwargs)
    table_feature = cf.get('table_feature', 0)
    if table_feature > 0:
        for key in ('num_features', 'num_tabular', 'in_features'):
            if key in mk:
                mk[key] = table_feature
        # Factories often omit these keys in registry defaults — force tabular dim.
        if model_name in ('hyperfusion', 'concat_fusion', 'cross_attention_fusion'):
            mk['num_tabular'] = table_feature
    if model_name.startswith('vit3d'):
        mk['img_size'] = tuple(cf.get('img_sz', (160, 160, 96)))
    return mk


def build_model(model_name, cf):
    """
    解析模型名与 config，返回:
      (model_cls, model_kwargs, forward_fn, use_score_prior)
    """
    if model_name in BACKBONE_MODELS:
        model_cls, kwargs = BACKBONE_MODELS[model_name]
        return model_cls, _inject_backbone_config(model_name, kwargs, cf), None, True

    if model_name in COMPARISON_MODELS:
        from training.trainer_comparison import comparison_forward_fn
        factory_fn, kwargs = COMPARISON_MODELS[model_name]
        mk = _inject_comparison_config(model_name, kwargs, cf)
        return type(factory_fn(**mk)), mk, comparison_forward_fn, False

    raise KeyError(model_name)


def print_model_list():
    print("\nAvailable models:")
    print(f"{'Model':<26} {'Type'}")
    print("-" * 40)
    for name in sorted(BACKBONE_MODELS):
        print(f"  {name:<24} backbone")
    for name in sorted(CAUSAL_MODELS):
        print(f"  {name:<24} causal")
    for name in sorted(COMPARISON_MODELS):
        print(f"  {name:<24} comparison")
    print()


def build_causal_model(model_name, cf, causal_phase=1):
    """Build disentangled / causal model with config injection."""
    if model_name not in CAUSAL_MODELS:
        raise KeyError(model_name)
    model_cls, kwargs = CAUSAL_MODELS[model_name]
    mk = _inject_backbone_config(model_name, kwargs, cf)
    mk['causal_phase'] = causal_phase
    return model_cls, mk


def build_age_confound_model(model_name, cf):
    """Build feature backbone for C2; falls back to BACKBONE_MODELS with get_feature=True."""
    try:
        return _build_age_confound_model_core(model_name, cf)
    except KeyError:
        if model_name in BACKBONE_MODELS:
            model_cls, kwargs = BACKBONE_MODELS[model_name]
            mk = _inject_backbone_config(model_name, kwargs, cf)
            mk['get_feature'] = True
            return model_cls, mk
        raise KeyError(f"Unknown age-confound model: {model_name}")


def print_causal_model_list():
    print("\nCausal models (phased construction M1–M5):")
    print(f"{'Model':<26} {'Default phase'}")
    print("-" * 40)
    for name in sorted(CAUSAL_MODELS):
        print(f"  {name:<24} 1")
    print()
