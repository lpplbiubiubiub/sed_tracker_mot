#coding:utf8
import visdom
import time
import numpy as np
import cv2


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d, k=None):
        self.vis.close(win=None)
        if type(d) == dict:
            for k, v in d.iteritems():
                self.img(k, v)
        elif type(d) == list:
            type_list = [type(x) == np.ndarray for x in d]
            if all(type_list):
                ims = [cv2.resize(im, (200, 200), interpolation=cv2.INTER_CUBIC) for im in d]
                ims = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in ims]
                ims = np.stack(ims)
                ims = ims.transpose(0, 3, 1, 2)
                if k is None:
                    self.vis.images(ims, win=str("sample rois"))
                else:
                    self.vis.images(ims, win=k)




    def img_seq(self, d, **kwargs):
        self.vis.close(win=None)
        for k, v in d.iteritems():
            if len(v) != 0:
                ims = [cv2.resize(im, (200, 200), interpolation=cv2.INTER_CUBIC) for im in v]
                ims = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in ims]
                ims = np.stack(ims)
                ims = ims.transpose(0, 3, 1, 2)
                self.vis.images(ims, win=str(k))
                # self.vis.video(ims, win=str(k))
        # self.vis.text("low level track has {}\n".format(len(d.keys())), win="log")
        if kwargs.has_key("track"):
            track_len = len(kwargs["track"])
            self.vis.text("there are {} tracks\n and track is {}".format(track_len, kwargs["track"]))


    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=unicode(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        '''
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        '''
        return getattr(self.vis, name)


if __name__ == "__main__":

    v = Visualizer("main")
    x = np.random.randint(0, 255, (3,3,100,100))
    v.vis.images(x, win="xxx")