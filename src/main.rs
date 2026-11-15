#![no_std]
#![no_main]
#![forbid(unsafe_code)]

mod dm_mc02;

use core::f32::consts::PI;
use core::sync::atomic::{AtomicU32, Ordering};
use dm_mc02::Can1;
use embassy_executor::Spawner;
use embassy_stm32::can::Frame;
use embassy_time::{Duration, Instant, Ticker};

// This rig uses GM6020 voltage commands for motor IDs 1 to 4.
const MOTOR_COMMAND_ID: u16 = 0x1FF;
// Fill these after reading the IDs on the teaching rig. Zero is intentionally invalid:
// before hardware inspection, feedback is ignored and no motor command is encoded.
const YAW_MOTOR_ID: u16 = 0;
const PITCH_MOTOR_ID: u16 = 0;
// Replace these after measuring the rig's mechanical reference pose.
const YAW_ZERO_RAD: f32 = 0.0;
const PITCH_ZERO_RAD: f32 = 0.0;
// Keep automatic recentering disabled until the two zero offsets and travel limits are known.
const AUTO_RECENTER_ENABLED: bool = false;

const REMOTE_SILENT_TICKS: u16 = 100;
const FEEDBACK_TIMEOUT_TICKS: u16 = 20;
const PREPARE_TIMEOUT_TICKS: u16 = 5_000;
const PREPARE_STABLE_TICKS: u16 = 200;
const STALL_LIMIT_TICKS: u16 = 500;
const OSCILLATION_WINDOW_TICKS: u16 = 1_000;
const OSCILLATION_LIMIT_TICKS: u16 = 500;
const TICK: Duration = Duration::from_millis(1);
const DT_S: f32 = 0.001;
const PI2: f32 = 2.0 * PI;
const ANGLE_SCALE: f32 = PI2 / 8192.0;
const DEADZONE: f32 = 0.05;
const MAX_TARGET_SPEED_RAD_S: f32 = 1.5;
const PITCH_LIMIT_RAD: f32 = 1.2;
const PREPARE_MAX_RPM: f32 = 80.0;
const PREPARE_MAX_OUTPUT: f32 = 2_000.0;
const MAX_VOLTAGE_RAW: f32 = 25_000.0;

/// Command frames dropped because the CAN transmit queue was full, for example while no
/// motor acknowledges frames.
pub static TX_FAILED: AtomicU32 = AtomicU32::new(0);

/// Feedback and voltage command for one motor.
#[derive(Default)]
pub struct Motor {
    /// Single-turn motor shaft angle in rad; turns are not counted.
    pub angle_rad: f32,
    pub speed_rpm: f32,
    /// Feedback current as sent on the wire. The motor manual gives no scale to amperes.
    pub current_raw: i16,
    /// Voltage command counts, not volts; protocol range +/-25000.
    pub command_voltage_raw: f32,
}

impl Motor {
    /// Clear the voltage command in RAM. The motor receives zero when
    /// `Control::transmit` next runs, within 1 ms.
    pub fn disable(&mut self) {
        self.command_voltage_raw = 0.0;
    }

    pub fn read(&mut self, data: &[u8; 8]) {
        let angle = u16::from_be_bytes([data[0], data[1]]);
        self.angle_rad = angle as f32 * ANGLE_SCALE;
        self.speed_rpm = i16::from_be_bytes([data[2], data[3]]) as f32;
        self.current_raw = i16::from_be_bytes([data[4], data[5]]);
    }
}

/// PID gains, plus the integral and history kept between ticks.
#[derive(Default)]
pub struct Pid {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub integral: f32,
    pub previous_error: f32,
    pub integral_limit: f32,
    pub maxout: f32,
    pub initialized: bool,
}

impl Pid {
    pub const fn new(kp: f32, ki: f32, kd: f32, integral_limit: f32, maxout: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            previous_error: 0.0,
            integral_limit,
            maxout,
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
        self.initialized = false;
    }

    pub fn calculate(&mut self, target: f32, measured: f32, dt_s: f32) -> f32 {
        let error = target - measured;
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return 0.0;
        }

        let derivative = if self.initialized {
            (error - self.previous_error) / dt_s
        } else {
            self.initialized = true;
            0.0
        };

        let old_integral = self.integral;
        let proposed_integral = (old_integral + self.ki * error * dt_s)
            .clamp(-self.integral_limit, self.integral_limit);
        let unclamped_with_integral = self.kp * error + proposed_integral + self.kd * derivative;

        // Conditional integration prevents the integral from growing farther into saturation.
        let saturated_high = unclamped_with_integral > self.maxout && error > 0.0;
        let saturated_low = unclamped_with_integral < -self.maxout && error < 0.0;
        if !(saturated_high || saturated_low) {
            self.integral = proposed_integral;
        }

        self.previous_error = error;
        (self.kp * error + self.integral + self.kd * derivative).clamp(-self.maxout, self.maxout)
    }
}

/// The gimbal's stage, which decides the allowed operations and outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Zero output. Waits for valid remote input and fresh feedback from both motors,
    /// then for a new enable request.
    Waiting,
    /// Recenters both axes to the reference pose with limited speed and output.
    /// Enters `Running` once both axes settle.
    Prepare,
    /// The left stick moves each axis's `target_angle_rad`, and its cascade tracks it.
    Running,
    /// Zero output until a manual recovery.
    Error,
}

/// One gimbal axis: its motor, targets and cascaded PIDs.
#[derive(Default)]
pub struct Axis {
    pub motor: Motor,
    /// Position loop target in rad.
    pub target_angle_rad: f32,
    /// Speed loop target in rpm, set from the position loop output.
    pub target_rpm: f32,
    /// Position loop: rad in, rpm out.
    pub position_pid: Pid,
    /// Speed loop: rpm in, voltage counts out.
    pub speed_pid: Pid,
    pub feedback_age: u16,
    pub prepare_stable_ticks: u16,
    pub stall_ticks: u16,
    pub oscillation_window_ticks: u16,
    pub oscillation_high_ticks: u16,
    pub oscillation_sign_changes: u16,
    pub last_speed_sign: i8,
}

/// State shared by every control tick.
pub struct Control {
    pub state: State,
    pub yaw: Axis,
    pub pitch: Axis,
    pub remote: Option<dm_mc02::Dr16>,
    pub remote_silent_ticks: u16,
    pub prepare_ticks: u16,
    pub last_right_switch: u8,
    pub last_left_switch: u8,
    pub enable_armed: bool,
    pub recovery_armed: bool,
}

/// Start in `Waiting` with no remote input.
/// Set the initial value of each field you add here, including the PID gains.
impl Default for Control {
    fn default() -> Self {
        Self {
            state: State::Waiting,
            yaw: Axis {
                position_pid: Pid::new(8.0, 0.0, 0.0, 100.0, 120.0),
                speed_pid: Pid::new(5.0, 0.5, 0.0, 2_000.0, 2_000.0),
                ..Axis::default()
            },
            pitch: Axis {
                position_pid: Pid::new(8.0, 0.0, 0.0, 100.0, 120.0),
                speed_pid: Pid::new(5.0, 0.5, 0.0, 2_000.0, 2_000.0),
                ..Axis::default()
            },
            remote: None,
            remote_silent_ticks: REMOTE_SILENT_TICKS,
            prepare_ticks: 0,
            last_right_switch: 0,
            last_left_switch: 0,
            enable_armed: false,
            recovery_armed: false,
        }
    }
}

impl Control {
    /// Clear both motors' voltage commands in RAM.
    pub fn disable_all(&mut self) {
        self.yaw.motor.disable();
        self.pitch.motor.disable();
        self.yaw.target_rpm = 0.0;
        self.pitch.target_rpm = 0.0;
    }

    /// Runs every 1 ms: read the remote and motor feedback, update the state, then
    /// compute and send the outputs. Missing remote input stops output; a latched Error is preserved.
    pub fn tick(&mut self, can: &mut Can1) {
        self.read_remote();
        self.read_feedback(can);
        self.update_mode();
        if self.remote.is_none() && self.state != State::Error {
            self.state = State::Waiting;
        }
        self.update();
        self.transmit(can);
    }

    /// Take the latest input published by the remote task. After an invalid frame, or
    /// `REMOTE_SILENT_TICKS` ticks without one, clear the commands; return to Waiting unless already in Error.
    pub fn read_remote(&mut self) {
        self.remote_silent_ticks = self.remote_silent_ticks.saturating_add(1);
        if let Some(remote) = dm_mc02::REMOTE.try_take() {
            self.remote = remote;
            if self.remote.is_some() {
                self.remote_silent_ticks = 0;
            } else {
                self.enable_armed = false;
                self.recovery_armed = false;
            }
        }
        if self.remote_silent_ticks >= REMOTE_SILENT_TICKS {
            self.remote = None;
            self.enable_armed = false;
            self.recovery_armed = false;
        }
        if self.remote.is_none() {
            self.disable_all();
            if self.state != State::Error {
                self.state = State::Waiting;
            }
        }
    }

    /// Read every queued CAN frame without waiting for more.
    pub fn read_feedback(&mut self, can: &mut Can1) {
        self.yaw.feedback_age = self.yaw.feedback_age.saturating_add(1);
        self.pitch.feedback_age = self.pitch.feedback_age.saturating_add(1);
        let receiver = can.reader();
        while let Ok(received) = receiver.try_receive() {
            match received {
                Ok(envelope) => {
                    let frame = envelope.frame;
                    if frame.data().len() != 8 {
                        continue;
                    }
                    let id = match frame.id() {
                        embedded_can::Id::Standard(id) => id.as_raw(),
                        embedded_can::Id::Extended(_) => continue,
                    };
                    let mut data = [0u8; 8];
                    data.copy_from_slice(frame.data());
                    if id == 0x204 + YAW_MOTOR_ID {
                        self.yaw.motor.read(&data);
                        self.yaw.motor.angle_rad = wrap_pi(self.yaw.motor.angle_rad - YAW_ZERO_RAD);
                        self.yaw.feedback_age = 0;
                    } else if id == 0x204 + PITCH_MOTOR_ID {
                        self.pitch.motor.read(&data);
                        self.pitch.motor.angle_rad =
                            wrap_pi(self.pitch.motor.angle_rad - PITCH_ZERO_RAD);
                        self.pitch.feedback_age = 0;
                    }
                }
                Err(error) => dm_mc02::log_can_error(error),
            }
        }
    }

    /// Choose the state from the remote, feedback and fault conditions.
    pub fn update_mode(&mut self) {
        let feedback_ready = self.yaw.feedback_age < FEEDBACK_TIMEOUT_TICKS
            && self.pitch.feedback_age < FEEDBACK_TIMEOUT_TICKS;
        let (right_switch, left_switch) = self
            .remote
            .map(|r| (r.right_switch, r.left_switch))
            .unwrap_or((0, 0));

        if right_switch == 2 {
            self.enable_armed = true;
        }
        if left_switch == 2 {
            self.recovery_armed = true;
        }

        match self.state {
            State::Waiting => {
                self.disable_all();
                if feedback_ready && self.remote.is_some() && right_switch == 3 && self.enable_armed
                {
                    self.enable_armed = false;
                    self.prepare_ticks = 0;
                    self.set_prepare_targets();
                    self.reset_axis_controllers();
                    self.state = State::Prepare;
                    rtt_target::rprintln!("state -> Prepare");
                }
            }
            State::Prepare => {
                if !feedback_ready || right_switch == 2 {
                    self.state = State::Waiting;
                    self.disable_all();
                    return;
                }
                self.prepare_ticks = self.prepare_ticks.saturating_add(1);
                if !AUTO_RECENTER_ENABLED {
                    self.yaw.target_angle_rad = self.yaw.motor.angle_rad;
                    self.pitch.target_angle_rad = self.pitch.motor.angle_rad;
                }
                let yaw_done =
                    angle_error(self.yaw.target_angle_rad, self.yaw.motor.angle_rad).abs() < 0.02
                        && self.yaw.motor.speed_rpm.abs() < 5.0;
                let pitch_done =
                    angle_error(self.pitch.target_angle_rad, self.pitch.motor.angle_rad).abs()
                        < 0.02
                        && self.pitch.motor.speed_rpm.abs() < 5.0;
                if yaw_done && pitch_done {
                    self.yaw.prepare_stable_ticks = self.yaw.prepare_stable_ticks.saturating_add(1);
                    self.pitch.prepare_stable_ticks =
                        self.pitch.prepare_stable_ticks.saturating_add(1);
                } else {
                    self.yaw.prepare_stable_ticks = 0;
                    self.pitch.prepare_stable_ticks = 0;
                }
                if self.yaw.prepare_stable_ticks >= PREPARE_STABLE_TICKS
                    && self.pitch.prepare_stable_ticks >= PREPARE_STABLE_TICKS
                {
                    self.reset_axis_controllers();
                    self.state = State::Running;
                    rtt_target::rprintln!("state -> Running");
                } else if self.prepare_ticks >= PREPARE_TIMEOUT_TICKS {
                    self.enter_error("prepare timeout");
                }
            }
            State::Running => {
                if !feedback_ready || right_switch == 2 {
                    self.disable_all();
                    if right_switch == 2 {
                        self.state = State::Waiting;
                        self.enable_armed = true;
                    } else {
                        self.state = State::Waiting;
                    }
                    return;
                }
                self.update_fault_detection();
            }
            State::Error => {
                self.disable_all();
                // Recovery gesture: left switch down, then up. A held position cannot recover.
                if self.remote.is_some()
                    && self.recovery_armed
                    && left_switch == 1
                    && right_switch == 3
                {
                    self.recovery_armed = false;
                    self.yaw.target_angle_rad = self.yaw.motor.angle_rad;
                    self.pitch.target_angle_rad = self.pitch.motor.angle_rad;
                    self.yaw.target_rpm = 0.0;
                    self.pitch.target_rpm = 0.0;
                    self.reset_axis_controllers();
                    self.reset_fault_detection();
                    self.state = State::Running;
                    rtt_target::rprintln!("state -> Running (manual recovery)");
                }
            }
        }
    }

    /// Set targets for the current state, run each axis's PIDs and write the voltage commands.
    pub fn update(&mut self) {
        match self.state {
            State::Waiting | State::Error => self.disable_all(),
            State::Prepare => {
                self.update_axis(true, PREPARE_MAX_RPM, PREPARE_MAX_OUTPUT, YAW_ZERO_RAD);
                self.update_axis_for_pitch(true, PREPARE_MAX_RPM, PREPARE_MAX_OUTPUT);
            }
            State::Running => {
                self.update_targets_from_remote();
                self.update_axis(false, 0.0, 0.0, 0.0);
                self.update_axis_for_pitch(false, 0.0, 0.0);
            }
        }
    }

    /// Queue one group command frame on CAN1 without waiting. This runs every tick, so a
    /// cleared command reaches the motor within 1 ms. A full queue drops the frame and
    /// counts it in `TX_FAILED`.
    pub fn transmit(&mut self, can: &mut Can1) {
        let mut data = [0u8; 8];
        put_voltage(&mut data, YAW_MOTOR_ID, self.yaw.motor.command_voltage_raw);
        put_voltage(
            &mut data,
            PITCH_MOTOR_ID,
            self.pitch.motor.command_voltage_raw,
        );
        let frame = Frame::new_standard(MOTOR_COMMAND_ID, &data).unwrap();
        if can.writer().try_write(frame).is_err() {
            TX_FAILED.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn update_axis(&mut self, prepare: bool, max_rpm: f32, max_output: f32, _zero: f32) {
        update_axis_control(&mut self.yaw, prepare, max_rpm, max_output, DT_S);
    }

    fn update_axis_for_pitch(&mut self, prepare: bool, max_rpm: f32, max_output: f32) {
        update_axis_control(&mut self.pitch, prepare, max_rpm, max_output, DT_S);
    }

    fn update_targets_from_remote(&mut self) {
        let Some(remote) = self.remote else {
            return;
        };
        let yaw_input = apply_deadzone(remote.channels[2] as f32 / 660.0);
        let pitch_input = apply_deadzone(remote.channels[3] as f32 / 660.0);
        self.yaw.target_angle_rad =
            wrap_pi(self.yaw.target_angle_rad + yaw_input * MAX_TARGET_SPEED_RAD_S * DT_S);
        self.pitch.target_angle_rad = (self.pitch.target_angle_rad
            + pitch_input * MAX_TARGET_SPEED_RAD_S * DT_S)
            .clamp(-PITCH_LIMIT_RAD, PITCH_LIMIT_RAD);
    }

    fn set_prepare_targets(&mut self) {
        if AUTO_RECENTER_ENABLED {
            self.yaw.target_angle_rad = 0.0;
            self.pitch.target_angle_rad = 0.0;
        } else {
            self.yaw.target_angle_rad = self.yaw.motor.angle_rad;
            self.pitch.target_angle_rad = self.pitch.motor.angle_rad;
        }
    }

    fn reset_axis_controllers(&mut self) {
        self.yaw.position_pid.reset();
        self.yaw.speed_pid.reset();
        self.pitch.position_pid.reset();
        self.pitch.speed_pid.reset();
        self.yaw.target_rpm = 0.0;
        self.pitch.target_rpm = 0.0;
    }

    fn reset_fault_detection(&mut self) {
        self.yaw.stall_ticks = 0;
        self.pitch.stall_ticks = 0;
        self.yaw.oscillation_window_ticks = 0;
        self.pitch.oscillation_window_ticks = 0;
        self.yaw.oscillation_high_ticks = 0;
        self.pitch.oscillation_high_ticks = 0;
        self.yaw.oscillation_sign_changes = 0;
        self.pitch.oscillation_sign_changes = 0;
        self.yaw.last_speed_sign = 0;
        self.pitch.last_speed_sign = 0;
    }

    fn update_fault_detection(&mut self) {
        if axis_has_fault(&mut self.yaw) {
            self.enter_error("yaw fault");
        } else if axis_has_fault(&mut self.pitch) {
            self.enter_error("pitch fault");
        }
    }

    fn enter_error(&mut self, reason: &str) {
        self.state = State::Error;
        self.disable_all();
        self.reset_axis_controllers();
        rtt_target::rprintln!("state -> Error: {}", reason);
    }
}

fn update_axis_control(axis: &mut Axis, prepare: bool, max_rpm: f32, max_output: f32, dt_s: f32) {
    let error = angle_error(axis.target_angle_rad, axis.motor.angle_rad);
    let mut target_rpm = axis.position_pid.calculate(error, 0.0, dt_s);
    if prepare {
        target_rpm = target_rpm.clamp(-max_rpm, max_rpm);
    }
    axis.target_rpm = target_rpm;
    if prepare {
        let old_limit = axis.speed_pid.maxout;
        axis.speed_pid.maxout = max_output;
        axis.motor.command_voltage_raw =
            axis.speed_pid
                .calculate(axis.target_rpm, axis.motor.speed_rpm, dt_s);
        axis.speed_pid.maxout = old_limit;
    } else {
        axis.motor.command_voltage_raw =
            axis.speed_pid
                .calculate(axis.target_rpm, axis.motor.speed_rpm, dt_s);
    }
}

fn axis_has_fault(axis: &mut Axis) -> bool {
    let demand = axis.target_rpm.abs() > 20.0;
    let high_output = axis.motor.command_voltage_raw.abs() > 1_000.0;
    if demand && high_output && axis.motor.speed_rpm.abs() < 5.0 {
        axis.stall_ticks = axis.stall_ticks.saturating_add(1);
    } else {
        axis.stall_ticks = 0;
    }

    axis.oscillation_window_ticks = axis.oscillation_window_ticks.saturating_add(1);
    if axis.motor.speed_rpm.abs() > 30.0 {
        let sign = if axis.motor.speed_rpm > 0.0 { 1 } else { -1 };
        if axis.last_speed_sign != 0 && sign != axis.last_speed_sign {
            axis.oscillation_sign_changes = axis.oscillation_sign_changes.saturating_add(1);
        }
        axis.last_speed_sign = sign;
    }
    if axis.oscillation_window_ticks >= OSCILLATION_WINDOW_TICKS {
        if axis.oscillation_sign_changes >= 8 {
            axis.oscillation_high_ticks = axis
                .oscillation_high_ticks
                .saturating_add(OSCILLATION_WINDOW_TICKS);
        } else {
            axis.oscillation_high_ticks = 0;
        }
        axis.oscillation_window_ticks = 0;
        axis.oscillation_sign_changes = 0;
    }
    axis.stall_ticks >= STALL_LIMIT_TICKS || axis.oscillation_high_ticks >= OSCILLATION_LIMIT_TICKS
}

fn apply_deadzone(input: f32) -> f32 {
    if input.abs() <= DEADZONE {
        0.0
    } else {
        let sign = if input > 0.0 { 1.0 } else { -1.0 };
        sign * ((input.abs() - DEADZONE) / (1.0 - DEADZONE))
    }
}

fn angle_error(target: f32, measured: f32) -> f32 {
    wrap_pi(target - measured)
}

fn wrap_pi(mut angle: f32) -> f32 {
    while angle > PI {
        angle -= PI2;
    }
    while angle < -PI {
        angle += PI2;
    }
    angle
}

fn put_voltage(data: &mut [u8; 8], motor_id: u16, value: f32) {
    if !(1..=4).contains(&motor_id) {
        return;
    }
    let value = value.clamp(-MAX_VOLTAGE_RAW, MAX_VOLTAGE_RAW) as i16;
    let bytes = value.to_be_bytes();
    let index = (motor_id as usize - 1) * 2;
    data[index] = bytes[0];
    data[index + 1] = bytes[1];
}

// Set up RTT and the board, then start the remote and control tasks.
#[embassy_executor::main]
async fn main(spawner: Spawner) {
    rtt_target::rtt_init_print!(rtt_target::ChannelMode::NoBlockTrim, 1024);
    let (can, watchdog, remote) = dm_mc02::init();
    rtt_target::rprintln!("Gimbal control started");
    spawner.spawn(dm_mc02::remote_task(remote).unwrap());
    spawner.spawn(control_task(can, watchdog).unwrap());
}

// Run `tick` at 1 kHz and feed the watchdog after each one. If a tick hangs, the watchdog
// resets the MCU after 100 ms.
#[embassy_executor::task]
async fn control_task(mut can: Can1, mut watchdog: dm_mc02::Watchdog) {
    let mut control = Control::default();
    let mut ticker = Ticker::every(TICK);
    let mut last = Instant::now();
    let mut log_ticks = 0u16;
    loop {
        ticker.next().await;
        let now = Instant::now();
        if now - last > TICK * 2 {
            ticker.reset();
        }
        last = now;
        control.tick(&mut can);
        log_ticks += 1;
        if log_ticks >= 200 {
            log_ticks = 0;
            rtt_target::rprintln!(
                "state={:?} yaw: angle={:.3} target={:.3} rpm={:.1} target_rpm={:.1} out={:.0}",
                control.state,
                control.yaw.motor.angle_rad,
                control.yaw.target_angle_rad,
                control.yaw.motor.speed_rpm,
                control.yaw.target_rpm,
                control.yaw.motor.command_voltage_raw,
            );
            rtt_target::rprintln!(
                "state={:?} pitch: angle={:.3} target={:.3} rpm={:.1} target_rpm={:.1} out={:.0}",
                control.state,
                control.pitch.motor.angle_rad,
                control.pitch.target_angle_rad,
                control.pitch.motor.speed_rpm,
                control.pitch.target_rpm,
                control.pitch.motor.command_voltage_raw,
            );
        }
        watchdog.pet();
    }
}
