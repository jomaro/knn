
extern crate rayon;
extern crate byteorder;
extern crate rulinalg;

use rulinalg::utils;

use rayon::prelude::*;

use std::env;
use std::fs::File;

use std::time::Instant;

use byteorder::{ReadBytesExt, LittleEndian};

use std::process::exit;


#[derive(Debug)]
struct Data {

	d : usize,
	c : usize,

	data : Vec<f64>,
	rotulos : Vec<i32>,
	test : Vec<f64>,
	rotulos_test : Vec<i32>,
}

//-> (i32, i32, i32, i32, Vec<f64>, Vec<i32>, Vec<f64>, Vec<i32>)
fn read_files(a : &str, b : &str) -> Data {
	let mut f = File::open(a).expect("Não abriu arquivo de dados");

	let n = f.read_i32::<LittleEndian>().expect("Não conseguiu ler N");

	let m = f.read_i32::<LittleEndian>().expect("Não conseguiu ler M");

	let d = f.read_i32::<LittleEndian>().expect("Não conseguiu ler D");

	let c = f.read_i32::<LittleEndian>().expect("Não conseguiu ler C");

	let mut data         = Vec::with_capacity((n*d) as usize);
	let mut rotulos      = Vec::with_capacity(n as usize);
	let mut test         = Vec::with_capacity((m*d) as usize);
	let mut rotulos_test = Vec::with_capacity(m as usize);


	data.resize((n*d) as usize, 0f64);
	rotulos.resize(n as usize, 0i32);

	test.resize((m*d) as usize, 0f64);
	rotulos_test.resize(m as usize, 0i32);

	//f.read(base.as_mut_slice() as &mut [u8]);
	//f.read(rotulos.as_mut_slice());
	f.read_f64_into::<LittleEndian>(data.as_mut_slice()).expect("deu problema 1");
	f.read_i32_into::<LittleEndian>(&mut rotulos).expect("deu problema 2");
	f.read_f64_into::<LittleEndian>(test.as_mut_slice()).expect("deu problema 3");

	let mut f = File::open(b).expect("Não abriu arquivo de teste");

	let m2 = f.read_i32::<LittleEndian>().expect("Os dados tem tamanhos diferentes");
	if m2 != m {
		panic!("Os dados tem tamanhos diferentes")
	};

	f.read_i32_into::<LittleEndian>(&mut rotulos_test).expect("não conseguiu ler rotulos");

	Data {
		d: d as usize,
		c: c as usize,
		data: data,
		rotulos: rotulos,
		test: test,
		rotulos_test: rotulos_test
	}
}


fn predict(x : &[f64], k : usize, data_ : &Data) -> i32 {

	let d = data_.d;
	let c = data_.c;

	let data = &data_.data;
	let rotulos = &data_.rotulos;


	let mut viz = Vec::<(f64, usize)>::with_capacity(k+1);

	for (linha, r) in data.chunks(d).zip(rotulos) {

		let dist = x.iter().zip(linha)
					.map(|(a, b)| (a-b)*(a-b))
					.sum();

		viz.push((dist, *r as usize));

		viz.sort_by(|a, b| a.partial_cmp(b).unwrap());

		viz.truncate(k);
	}

	let mut votos = Vec::with_capacity(c);

	votos.resize(c, 0);

	for (_, v) in viz {
		votos[v] += 1;
	}

	utils::argmax(&votos).0 as i32
}

fn main(){
	let args: Vec<String> = env::args().collect();

	if args.len() != 4 {
		println!("Usage: ./knn test1.knn 5 test1-k5.knn");

		exit(1);
	}

	let data = read_files(&args[1], &args[3]);

	let k : usize = args[2].parse().expect("Couldn't parse K");

	let now = Instant::now();

	let rotulos_tests : Vec<i32> = data.test
		.chunks(data.d)
		.map(|x| predict(x, k, &data))
		.collect();

	println!("Sequencial: {:?}", Instant::now().duration_since(now));

	let missmatches_s = data.rotulos_test.iter()
		.zip(rotulos_tests)
		.filter(|&(a, b)| *a != b)
		.count();


	let now = Instant::now();

	let rotulos_testp : Vec<i32> = data.test.as_slice()
		.par_chunks(data.d)
		.map(|x| predict(x, k, &data))
		.collect();

	println!("Paralelo: {:?}", Instant::now().duration_since(now));

	let missmatches_p = data.rotulos_test.iter()
		.zip(rotulos_testp)
		.filter(|&(a, b)| *a != b)
		.count();

	println!("Serial   {} missmatches from {}", missmatches_s, data.rotulos_test.len());

	println!("Parallel {} missmatches from {}", missmatches_p, data.rotulos_test.len());
}
