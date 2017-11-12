"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from spaces import TorchContinuous
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from C_alpha_protein.Angles2CoordsAB import cppAngles2CoordsAB
from C_alpha_protein.Coords2Pairs import cppCoords2Pairs

class ABModel:
	def __init__(self, angles_length, interacton, batch_size=10):
		self.batch_size = batch_size
		self.angles_length = angles_length
		self.num_atoms = angles_length + 1
		
		self.A = torch.FloatTensor(batch_size, 16*self.angles_length).cuda()
		self.angles_length_tensor = torch.IntTensor(batch_size).fill_(self.angles_length)
		self.coords = torch.FloatTensor(batch_size, 3*self.num_atoms).cuda()
		self.pairs = torch.FloatTensor(batch_size, 3,self.num_atoms,self.num_atoms).cuda()

		self.mask = torch.ByteTensor(batch_size, self.num_atoms*self.num_atoms).cuda().fill_(0)
		self.mask[:, ::self.num_atoms+1]=1
		self.mask[:, 1::self.num_atoms+1]=1
		self.mask[:, self.num_atoms::self.num_atoms+1]=1
		self.mask.resize_(batch_size, self.num_atoms, self.num_atoms)
		
		self.interaction = interaction

	def step(self, angles):
		
		self.coords.fill_(0.0)
		cppAngles2CoordsAB.Angles2Coords_forward(   angles,              #input angles
													self.coords,  #output coordinates
													self.angles_length_tensor, 
													self.A)
		if math.isnan(self.coords.sum()):
			raise(Exception('ABModel: angles2coords forward Nan'))		

		self.pairs.fill_(0.0)
		
		cppCoords2Pairs.Coords2Pairs_forward( 	self.coords, #input coordinates
												self.pairs,  #output pairwise coordinates
												self.angles_length_tensor)
		if math.isnan(self.pairs.sum()):
			raise(Exception('ABModel: angles2coords forward Nan'))

	def energy(self, angles):
		
		V1 = self.angles_length - torch.sum(torch.cos(angles[:,0,1:]), dim=1) - 1
		
		c_pairs = self.pairs.resize_(self.batch_size, 3, self.num_atoms, self.num_atoms)
		r2 = torch.pow( c_pairs[:,0,:,:], 2) + torch.pow( c_pairs[:,1,:,:], 2) + torch.pow( c_pairs[:,2,:,:], 2)
		r6 = 1.0/torch.pow(r2, 3)
		r6.masked_fill_(self.mask, 0.0)
		r12 = torch.pow(r6, 2)
		r12 = torch.addcmul(r12, value=-1, tensor1=r6, tensor2=self.interaction)
		V2 = torch.sum(torch.sum(r12,1),1)
		return 0.25*V1+2.0*V2
		

	def move_naive(self, angles):
		for i in xrange(self.batch_size):
			angle_index = random.randint(0, self.angles_length-1)
			angles[i,:,angle_index] = 2.0*math.pi*(torch.rand(2).cuda()-1.0)
		return angles

	def move_spherical(self, angles):
		for i in xrange(self.batch_size):
			angle_index = random.randint(0, self.angles_length-2)
			rand_rot = 0.1*2.0*math.pi*(torch.rand(2).cuda()-1.0)
			angles[i,:,angle_index] += rand_rot
			angles[i,:,angle_index+1] -= rand_rot
		return angles


class ABModelEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')