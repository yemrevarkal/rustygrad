use crate::node::{NodeRef, Node};
use rand::Rng;

#[derive(Debug)]
pub struct Neuron{
    pub weights: Vec<NodeRef>,
    pub bias: NodeRef,
    pub nonlin: bool
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut weights = vec![];
        for _ in 0..nin {
            weights.push(Node::new(rng.gen_range(-1.0..1.0)));
        }
        let bias = Node::new(0.0);
        Neuron {
            weights,
            bias,
            nonlin
        }
    }
    
    pub fn forward(&mut self, x: Vec<NodeRef>) -> NodeRef{
        let mut sum = Node::new(0.0); 
        for (wi,xi) in self.weights.iter_mut().zip(x.into_iter()) {
            sum = sum + wi.clone()*xi 
        }
        if self.nonlin {
            let act = (sum + self.bias.clone()).relu();
            act
        } else {
            let act = sum + self.bias.clone();
            act
        }
    }

    pub fn parameters(&self) -> Vec<&NodeRef> {
        let mut out = vec![&self.bias];
        out.extend(&self.weights);
        out
    }

    pub fn update(&self, lr: f64) {
        for w in self.parameters() {
            let grad = w.0.borrow()._backward.get_output_immut().grad;
            w.0.borrow_mut()._backward.get_output().value -= lr*grad;

            {
                w.0.borrow_mut()._backward.get_output().grad = 0.0; // reset grad
            }
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin:bool) -> Layer {
        let mut neurons = vec![];
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, nonlin)); 
        }

        Layer {
            neurons,
        }
    }

    pub fn forward(&mut self, x: Vec<NodeRef> ) -> Vec<NodeRef> {
        let mut out = vec![];

        for n in self.neurons.iter_mut() {
            out.push(n.forward(x.clone()));
        };

        out
    }

    pub fn parameters(&self) -> Vec<&NodeRef> {
        let mut out = vec![];
        for n in self.neurons.iter() {
            out.extend(n.parameters());
            
        }
        out
    }

    pub fn update(&self, lr: f64) {
        for n in self.neurons.iter() {
            n.update(lr);
        }
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> MLP {
        let mut size = vec![nin];
        size.extend(nouts.clone());
        let mut layers = vec![];
        
        for i in 0..nouts.len() {
            let nonlin = i == nouts.len(); // last layer is non-linear
            layers.push(Layer::new(size[i],size[i+1],nonlin));
        }

        MLP {
            layers
        }
    }

    pub fn forward(&mut self, x: Vec<NodeRef>) -> Vec<NodeRef> {
        let mut input_x = x.clone();
        for l in self.layers.iter_mut() {
            input_x = l.forward(input_x) 
        }
        input_x
    }

    pub fn parameters(&self) -> Vec<&NodeRef> {
        let mut out = vec![];
        for l in self.layers.iter() {
            out.extend(l.parameters());
        }
        out
    }

    pub fn update(&self, lr: f64) {
        for l in self.layers.iter() {
            l.update(lr);
        }
    }
}


impl MLP {
    pub fn forward_vec(&mut self, x: Vec<Vec<f64>>) -> Vec<NodeRef> {
        let mut res = vec![];

        for v in x {
            let input_x = v.iter().map(|x| Node::new(*x)).collect();
            let out = self.forward(input_x);
            res.push(out[0].clone());
        }
        res
    }
}


pub struct MSELoss {
    pub y_true: Vec<f64>,
    pub y_pred: Vec<NodeRef>
}

impl MSELoss {
    pub fn loss(&self) -> NodeRef {
        let mut loss = Node::new(0.0);
        for (y_true, y_pred) in self.y_true.iter().zip(self.y_pred.iter()) {
            let lossi = (Node::new(y_true.clone())+Node::new(-1.0)*y_pred.clone()).pow(2.0);
            loss = loss + lossi;
        }
        loss
    }
}
