use std::f64;
use std::fmt::Debug;
use std::ops::{Mul, Add};

use std::rc::Rc;
use std::cell::RefCell;

use std::collections::HashSet;
use dyn_clone::DynClone;

pub trait Backward: std::fmt::Debug + DynClone{
    fn backward(&mut self) -> f64; 
    fn get_children(&mut self) -> &mut Vec<NodeRef>;
    fn get_output(&mut self) -> &mut ValueOutput;
    fn get_output_immut(&self) -> &ValueOutput;
}
dyn_clone::clone_trait_object!(Backward);

#[derive(Debug, Clone)]
pub struct ValueOutput { 
    pub value: f64,
    pub grad: f64
}

#[derive(Debug, Clone)]
pub struct Node {
    pub _backward: Box<dyn Backward>,
    pub _op: String 
    
}

#[derive(Debug, Clone)]
pub struct NodeRef(pub Rc<RefCell<Node>>);

#[derive(Debug, Clone)]
struct DefaultBackward {
    _children: Vec<NodeRef>,
    output: ValueOutput
}

impl Backward for DefaultBackward {
    fn backward(&mut self) -> f64 {
        1.0
    }
    fn get_children(&mut self) -> &mut Vec<NodeRef> {
        &mut self._children   
    }
    fn get_output(&mut self) -> &mut ValueOutput {
        &mut self.output   
    }
    fn get_output_immut(&self) -> &ValueOutput {
        &self.output   
    }
}

impl Node {
    pub fn new(value: f64) -> NodeRef  {
        
        let output = ValueOutput {
            value,
            grad: 0.0
        };
        let _children = vec![];

        let out = Node {
            _backward: Box::new(DefaultBackward{_children,output}),
            _op: String::from("")
        };
        NodeRef(Rc::new(RefCell::new(out)))
    }
}

#[derive(Debug, Clone)]
struct AddBackward {
    _children: Vec<NodeRef>,
    output: ValueOutput
}
impl Backward for AddBackward {

   fn backward(&mut self) -> f64 {
        let mut child0 = self._children[0].0.borrow_mut();
        let mut child1 = self._children[1].0.borrow_mut();

        child0._backward.get_output().grad += self.output.grad;
        child1._backward.get_output().grad += self.output.grad;
        1.0
    }
  
    fn get_children(&mut self) -> &mut Vec<NodeRef> {
        &mut self._children   
    }
    fn get_output(&mut self) -> &mut ValueOutput {
        &mut self.output   
    }
    fn get_output_immut(&self) -> &ValueOutput {
        &self.output   
    }
}

impl Add for NodeRef {
    type Output = NodeRef;

    fn add(self, other: NodeRef) -> NodeRef {
        let output = ValueOutput {
            value: self.0.borrow()._backward.get_output_immut().value + other.0.borrow()._backward.get_output_immut().value,
            grad: 0.0
        }; 
    
        let _children = vec![self, other];
        let out = Node {
            _backward: Box::new(AddBackward{_children, output}),
            _op: String::from("+")
        };
        NodeRef(Rc::new(RefCell::new(out)))
    }

}


#[derive(Debug, Clone)]
struct MulBackward {
    _children: Vec<NodeRef>,
    output: ValueOutput
}
impl Backward for MulBackward {

   fn backward(&mut self) -> f64 {
        let mut child0 = self._children[0].0.borrow_mut();
        let mut child1 = self._children[1].0.borrow_mut();

        child0._backward.get_output().grad += child1._backward.get_output_immut().value*self.output.grad;
        child1._backward.get_output().grad += child0._backward.get_output_immut().value*self.output.grad;
        1.0
    }
  
    fn get_children(&mut self) -> &mut Vec<NodeRef> {
        &mut self._children   
    }
    fn get_output(&mut self) -> &mut ValueOutput {
        &mut self.output   
    }
    fn get_output_immut(&self) -> &ValueOutput {
        &self.output   
    }
}

impl Mul for NodeRef {
    type Output = NodeRef;

    fn mul(self, other: NodeRef) -> NodeRef {
        let output = ValueOutput {
            value: self.0.borrow()._backward.get_output_immut().value * other.0.borrow()._backward.get_output_immut().value,
            grad: 0.0
        }; 
    
        let _children = vec![self, other];
        let out = Node {
            _backward: Box::new(MulBackward{_children, output}),
            _op: String::from("+")
        };
        NodeRef(Rc::new(RefCell::new(out)))
    }

}

#[derive(Debug, Clone)]
struct PowBackward {
    _children: Vec<NodeRef>,
    output: ValueOutput,
    power: f64
}
impl Backward for PowBackward {

   fn backward(&mut self) -> f64 {
        let mut child0 = self._children[0].0.borrow_mut();

        child0._backward.get_output().grad += self.power*f64::powf(child0._backward.get_output_immut().value, self.power-1.0)*self.output.grad;
        1.0
    }
  
    fn get_children(&mut self) -> &mut Vec<NodeRef> {
        &mut self._children   
    }
    fn get_output(&mut self) -> &mut ValueOutput {
        &mut self.output   
    }
    fn get_output_immut(&self) -> &ValueOutput {
        &self.output   
    }
}


impl NodeRef {
    pub fn pow(self, power: f64) -> NodeRef {
        let output = ValueOutput {
            value: f64::powf(self.0.borrow()._backward.get_output_immut().value, power),
            grad: 0.0
        }; 
    
        let _children = vec![self];
        let out = Node {
            _backward: Box::new(PowBackward{_children, output, power}),
            _op: String::from("+")
        };
        NodeRef(Rc::new(RefCell::new(out)))
    }

}

#[derive(Debug, Clone)]
struct ReluBackward {
    _children: Vec<NodeRef>,
    output: ValueOutput,
}
impl Backward for ReluBackward {

   fn backward(&mut self) -> f64 {
        let mut child0 = self._children[0].0.borrow_mut();

        child0._backward.get_output().grad += f64::from(self.output.grad>0.0)*self.output.grad;
        1.0
    }
  
    fn get_children(&mut self) -> &mut Vec<NodeRef> {
        &mut self._children   
    }
    fn get_output(&mut self) -> &mut ValueOutput {
        &mut self.output   
    }
    fn get_output_immut(&self) -> &ValueOutput {
        &self.output   
    }
}

impl NodeRef {
    pub fn relu(self) -> NodeRef {
        let output = ValueOutput {
            value: f64::max(0.0, self.0.borrow()._backward.get_output_immut().value),
            grad: 0.0
        }; 
    
        let _children = vec![self];
        let out = Node {
            _backward: Box::new(ReluBackward{_children, output}),
            _op: String::from("+")
        };
        NodeRef(Rc::new(RefCell::new(out)))
    }

}

fn topo_backward<'a>(v: &mut NodeRef, visited: &mut HashSet<usize>) {
    let ptr = v as *const NodeRef as usize;
    if !visited.contains(&ptr) {
        //println!("{:?} {} {:?}", ptr, v.0.borrow()._backward.get_output_immut().value, v.0.borrow()._backward.get_output_immut().grad);
        visited.insert(ptr);
        v.0.borrow_mut()._backward.backward();
        for child in v.0.borrow_mut()._backward.get_children() {
            topo_backward(child, visited);
        }
        
    }
}

impl NodeRef
{
    pub fn backward(&mut self) {
        let mut _visited = HashSet::new();
        self.0.borrow_mut()._backward.get_output().grad = 1.0;
        topo_backward(self, &mut _visited);
    }
}


