mod node;
mod nn;

use node::Node;
use csv::ReaderBuilder;
use std::fs::File;


fn load_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("src/synthetic_data.csv")
        .unwrap();

    let mut input_vec = Vec::new();
    let mut result_vec = Vec::new();
    for result in rdr.records() {
        let result = result.unwrap();
        let mut input = Vec::new();
        for (i, r) in result.iter().enumerate() {
            if i == result.len()-1 {
                result_vec.push(r.parse::<f64>().unwrap());
            } else {
                input.push(r.parse::<f64>().unwrap());
            }
        }
        input_vec.push(input);
    }

    (input_vec, result_vec)

}

fn test(a:f64) -> f64 {
    let nd1 = Node::new(12.0); 
    let nd2 = Node::new(5.0); 
    let nd3 = Node::new(7.0); 

    let nd4 = nd1+nd2+nd3; 
    let nd5 = Node::new(7.0+a)*nd4; 
    let nd5_1 = nd5*Node::new(2.0); 
    let mut nd6 = nd5_1.pow(2.0); 
     
    
    nd6.backward();
    println!("\n ------- \n");
    let val = nd6.0.borrow()._backward.get_output_immut().value;
    val
}


fn main() {
   
    let a = 0.000000001;
    let t1 = test(0.0);
    let t2 = test(a);
    println!("t1: {}", t1);
    println!("t2: {}", t2);
    println!("derivative: {}", (t2-t1)/a);

    let mut mlp = nn::MLP::new(3, vec![8,8, 1]);
    
    // let input_vec = vec![
    //     vec![1.0, 4.0, 5.0],
    //     vec![0.0, 2.0, 3.0],
    //     vec![5.0, 10.0, 5.0],
    //     vec![5.0, 2.0, 3.0],
    // ];
    // let result_vec = vec![2.7, 2.0, 3.0, 4.0];

    let data = load_data();
    let input_vec = data.0;
    let result_vec = data.1;
    println!("input_vec: {}", input_vec.len());
    
    let epoch = 1000;
    for i in 0..epoch {
        let y_pred = mlp.forward_vec(input_vec.clone()); 
        let mse = nn::MSELoss {
            y_true: result_vec.clone(),
            y_pred: y_pred.clone()
        };

        let mut loss = mse.loss();
        loss.backward();
        mlp.update(0.00001);

        println!("loss: {} grad: {}", loss.0.borrow()._backward.get_output_immut().value, loss.0.borrow()._backward.get_output_immut().grad);
        // for param in mlp.parameters().iter() {
        //     println!("param: {} grad: {}", param.0.borrow()._backward.get_output_immut().value, param.0.borrow()._backward.get_output_immut().grad);
        // }
        
        if i == epoch-1 {
            for (yp, yt) in y_pred.iter().zip(result_vec.iter()) {
                println!("yp: {} yt: {}", yp.0.borrow()._backward.get_output_immut().value, yt);
            }

            // for p in mlp.parameters().iter() {
            //     println!("param: {}", p.0.borrow()._backward.get_output_immut().value);
            // }
        }
    }
}
