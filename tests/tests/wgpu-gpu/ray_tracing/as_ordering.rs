use std::borrow::Cow;

use super::{acceleration_structure_limits, AsBuildContext};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(BUILD_QUERY_SUBMISSION_ORDER);
}

#[apply(gpu_test!)]
static BUILD_QUERY_SUBMISSION_ORDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(build_query_submission_order);

#[derive(Clone, Copy, Debug)]
enum SubmissionOrder {
    SplitEncoders,
    DistinctSubmits,
    ReverseRecording,
}

const SHADER: &str = r#"
enable wgpu_ray_query;
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> hits: array<vec4u>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) index: u32) {
    var x = 0.0;
    if (index == 1u) { x = 3.0; }
    if (index == 2u) { x = 8.0; }
    var query: ray_query;
    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0.0,
        select(10.0, 0.5, index == 3u), vec3f(x, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
    while (rayQueryProceed(&query)) {
        if (rayQueryGetCandidateIntersection(&query).kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let hit = rayQueryGetCommittedIntersection(&query);
    hits[index] = vec4u(hit.kind, hit.instance_custom_data, bitcast<u32>(hit.t), hit.instance_index);
}
"#;

async fn build_query_submission_order(ctx: TestingContext) {
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
    });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut failures = Vec::new();
    for order in [
        SubmissionOrder::SplitEncoders,
        SubmissionOrder::DistinctSubmits,
        SubmissionOrder::ReverseRecording,
    ] {
        for iteration in 0..8u32 {
            let bottoms: Vec<_> = (0..2)
                .map(|_| {
                    let mut bottom = AsBuildContext::new(
                        &ctx,
                        AccelerationStructureFlags::empty(),
                        AccelerationStructureFlags::empty(),
                    );
                    bottom.vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[
                            [-1.0f32, -1.0, 0.0],
                            [1.0, -1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ]),
                        usage: BufferUsages::BLAS_INPUT,
                    });
                    bottom
                })
                .collect();
            let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
                label: None,
                max_instances: 2,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            });
            for (index, bottom) in bottoms.iter().enumerate() {
                let distance = 1.0 + index as f32 + iteration as f32 / 32.0;
                tlas[index] = Some(TlasInstance::new(
                    &bottom.blas,
                    [
                        1.0,
                        0.0,
                        0.0,
                        index as f32 * 3.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        distance,
                    ],
                    100 + iteration * 2 + index as u32,
                    255,
                ));
            }
            let output = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::AccelerationStructure(&tlas),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let record_query = || {
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 64);
                encoder.finish()
            };
            // Recording order must not impose dependencies on the eventual submission order.
            let recorded_query =
                matches!(order, SubmissionOrder::ReverseRecording).then(record_query);
            let mut blas_encoder = ctx.device.create_command_encoder(&Default::default());
            for bottom in &bottoms {
                blas_encoder.build_acceleration_structures([&bottom.blas_build_entry()], []);
            }
            let mut tlas_encoder = ctx.device.create_command_encoder(&Default::default());
            tlas_encoder.build_acceleration_structures([], [&tlas]);
            let query = recorded_query.unwrap_or_else(record_query);
            if matches!(order, SubmissionOrder::DistinctSubmits) {
                ctx.queue.submit([blas_encoder.finish()]);
                ctx.queue.submit([tlas_encoder.finish()]);
                ctx.queue.submit([query]);
            } else {
                ctx.queue
                    .submit([blas_encoder.finish(), tlas_encoder.finish(), query]);
            }
            readback.slice(..).map_async(MapMode::Read, Result::unwrap);
            ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
            let mapped = readback.slice(..).get_mapped_range().unwrap();
            let records: &[[u32; 4]] = bytemuck::cast_slice(&mapped);
            for (index, hit) in records.iter().enumerate() {
                if index >= 2 {
                    if hit[0] != 0 {
                        failures.push(format!(
                            "{order:?} fresh{iteration} ray{index}: expected miss, got {hit:?}"
                        ));
                    }
                    continue;
                }
                let distance = f32::from_bits(hit[2]);
                let expected_distance = 1.0 + index as f32 + iteration as f32 / 32.0;
                if hit[0] != 1
                    || hit[1] != 100 + iteration * 2 + index as u32
                    || hit[3] != index as u32
                    || !distance.is_finite()
                    || (distance - expected_distance).abs() >= 2e-5
                {
                    failures.push(format!("{order:?} fresh{iteration} ray{index}: {hit:?}, distance {distance}, expected {expected_distance}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
