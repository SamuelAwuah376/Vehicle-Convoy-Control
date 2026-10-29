"""Numpy re-implementation of the saved MaskablePPO actor (deterministic) and VecNormalize obs scaling (no torch)."""
import zipfile, pickle, io, types, sys
import numpy as np

class _Storage:
    def __init__(self, key, dtype): self.key, self.dtype = key, dtype

def load_state_dict(model_zip):
    z = zipfile.ZipFile(model_zip); inner = zipfile.ZipFile(io.BytesIO(z.read('policy.pth')))
    root = inner.namelist()[0].split('/')[0]
    dtmap = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64}
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2':
                def rb(storage, offset, size, stride, *a):
                    raw = np.frombuffer(inner.read(f'{root}/data/{storage.key}'), storage.dtype)
                    if not size: return raw[offset:offset+1].reshape(())
                    return np.lib.stride_tricks.as_strided(raw[offset:], size, [s*raw.itemsize for s in stride]).copy()
                return rb
            if mod == 'torch' and name.endswith('Storage'):
                return dtmap[name]
            if mod == 'collections' and name == 'OrderedDict':
                import collections; return collections.OrderedDict
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, cls, key, loc, n = pid
            return _Storage(key, cls)
    return U(io.BytesIO(inner.read(f'{root}/data.pkl'))).load()

class _Any:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {})

def load_vecnorm(path):
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(('stable_baselines3', 'sb3_contrib', 'gymnasium')):
                return _Any
            return super().find_class(mod, name)
    return U(open(path, 'rb')).load()

class NumpyPolicy:
    def __init__(self, model_zip, vecnorm_pkl, nvec=(45, 45, 45)):
        sd = load_state_dict(model_zip)
        self.sd = {k: np.asarray(v, np.float64) for k, v in sd.items()}
        self.layers = sorted({k.rsplit('.', 1)[0] for k in sd if k.startswith('mlp_extractor.policy_net')})
        vn = load_vecnorm(vecnorm_pkl)
        self.mean, self.var = np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var)
        self.clip, self.eps, self.norm = vn.clip_obs, vn.epsilon, vn.norm_obs
        self.nvec = nvec
    def __call__(self, obs, mask):
        x = np.asarray(obs, np.float64)
        if self.norm:
            x = np.clip((x-self.mean)/np.sqrt(self.var+self.eps), -self.clip, self.clip).astype(np.float32).astype(np.float64)
        for l in self.layers:
            x = np.tanh(self.sd[l+'.weight']@x+self.sd[l+'.bias'])
        logits = self.sd['action_net.weight']@x+self.sd['action_net.bias']
        mask = np.asarray(mask, bool); out, o = [], 0
        for n in self.nvec:
            lg = np.where(mask[o:o+n], logits[o:o+n], -1e8); out.append(int(np.argmax(lg))); o += n
        return np.array(out)
