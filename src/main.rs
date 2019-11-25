
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
struct PredictorData {

	line_length : usize,
	max_label : usize,

	given_points : Vec<f64>,
	given_labels : Vec<u32>,
	points_to_process : Vec<f64>,
	benchmark_labels : Vec<u32>,
}

fn read_files(know_points_filename : &str, classified_labels_filename : &str) -> PredictorData {
	let mut f = File::open(know_points_filename).expect("Não abriu arquivo de dados");

	let n = f.read_u32::<LittleEndian>().expect("Não conseguiu ler N");

	let m = f.read_u32::<LittleEndian>().expect("Não conseguiu ler M");

	let d = f.read_u32::<LittleEndian>().expect("Não conseguiu ler D");

	let c = f.read_u32::<LittleEndian>().expect("Não conseguiu ler C");

	let mut given_points      = Vec::with_capacity((n*d) as usize);
	let mut given_labels      = Vec::with_capacity(n as usize);
	let mut points_to_process = Vec::with_capacity((m*d) as usize);
	let mut benchmark_labels  = Vec::with_capacity(m as usize);


	given_points.resize((n*d) as usize, 0f64);
	given_labels.resize(n as usize, 0u32);

	points_to_process.resize((m*d) as usize, 0f64);
	benchmark_labels.resize(m as usize, 0u32);

	//f.read(base.as_mut_slice() as &mut [u8]);
	//f.read(rotulos.as_mut_slice());
	f.read_f64_into::<LittleEndian>(given_points.as_mut_slice()).expect("deu problema 1");
	f.read_u32_into::<LittleEndian>(&mut given_labels).expect("deu problema 2");
	f.read_f64_into::<LittleEndian>(points_to_process.as_mut_slice()).expect("deu problema 3");

	let mut f = File::open(classified_labels_filename).expect("Não abriu arquivo de teste");

	let m2 = f.read_u32::<LittleEndian>().expect("Os dados tem tamanhos diferentes");
	if m2 != m {
		panic!("Os dados tem tamanhos diferentes")
	};

	f.read_u32_into::<LittleEndian>(&mut benchmark_labels).expect("não conseguiu ler rotulos");

	PredictorData {
		line_length: d as usize,
		max_label: c as usize,
		given_points: given_points,
		given_labels: given_labels,
		points_to_process: points_to_process,
		benchmark_labels: benchmark_labels
	}
}


fn predict(point_to_classify : &[f64], k : usize, data_ : &PredictorData) -> u32 {

	let line_length = data_.line_length;
	let max_label = data_.max_label;

	let given_points = &data_.given_points;
	let given_labels = &data_.given_labels;


	let mut viz = Vec::<(f64, usize)>::with_capacity(k+1);

	for (point, point_label) in given_points.chunks(line_length).zip(given_labels) {

		let dist = point_to_classify.iter().zip(point)
					.map(|(a, b)| (a-b)*(a-b))
					.sum();

		viz.push((dist, *point_label as usize));

		viz.sort_by(|a, b| a.partial_cmp(b).unwrap());

		viz.truncate(k);
	}

	let mut votes = Vec::with_capacity(max_label);

	votes.resize(max_label, 0);

	for (_, v) in viz {
		votes[v] += 1;
	}

	utils::argmax(&votes).0 as u32
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

	let rotulos_tests : Vec<u32> = data.points_to_process
		.chunks(data.line_length)
		.map(|x| predict(x, k, &data))
		.collect();

	println!("Serial: {:?}", Instant::now().duration_since(now));

	let missmatches_s = data.benchmark_labels.iter()
		.zip(rotulos_tests)
		.filter(|&(a, b)| *a != b)
		.count();


	let now = Instant::now();

	let rotulos_testp : Vec<u32> = data.points_to_process.as_slice()
		.par_chunks(data.line_length)
		.map(|x| predict(x, k, &data))
		.collect();

	println!("Parallel: {:?}", Instant::now().duration_since(now));

	let missmatches_p = data.benchmark_labels.iter()
		.zip(rotulos_testp)
		.filter(|&(a, b)| *a != b)
		.count();

	println!("Serial   {} missmatches from {}", missmatches_s, data.benchmark_labels.len());

	println!("Parallel {} missmatches from {}", missmatches_p, data.benchmark_labels.len());
}
