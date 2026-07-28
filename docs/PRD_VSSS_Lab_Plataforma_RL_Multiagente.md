# PRD: VSSS Lab — Plataforma modular de simulación, RL multiagente y competencia

**Versión:** 1.0  
**Estado:** Ready for implementation  
**Fecha:** 27 de julio de 2026  
**Owner:** Roberto Villegas  
**Colaborador inicial:** Julio / autor de `simulation_vsss`  
**Repositorio propuesto:** `RobertoVillegas/vsss-lab`  
**Plataforma principal validada:** Windows 11 + WSL2 Ubuntu 26.04 + Docker Desktop + NVIDIA RTX 3070  
**Licencia propuesta:** Apache-2.0 para código propio; preservar licencias y atribuciones de assets y componentes derivados.

---

## 1. Resumen ejecutivo

VSSS Lab será una plataforma de investigación y competencia para Very Small Size Soccer. No será únicamente un simulador, un proyecto ROS ni un script de entrenamiento. Será un conjunto de contratos, motores y herramientas intercambiables para:

- simular partidos VSSS a alta velocidad;
- definir tareas de aprendizaje reproducibles;
- entrenar políticas single-agent y multi-agent;
- permitir políticas compartidas sin roles ligados a la identidad física del robot;
- ejecutar self-play, ligas y torneos;
- comparar controladores heurísticos, RL y combinaciones híbridas;
- validar las políticas en un simulador robótico de mayor fidelidad;
- preparar posteriormente integración con visión y robots físicos;
- permitir que agentes de desarrollo modifiquen el sistema de extremo a extremo sin depender de una infraestructura opaca.

La arquitectura separará cinco planos:

```text
1. Especificación canónica:
   reglas, geometría, estados, acciones, eventos y replay.

2. Simulación:
   backend CPU headless rápido, backends experimentales y backend ROS/Gazebo.

3. Entorno RL:
   observaciones, acciones, recompensas, resets, terminaciones y randomización.

4. Aprendizaje:
   IPPO, MAPPO, políticas compartidas, críticos centralizados, curriculum y self-play.

5. Competencia:
   match server, protocolo externo, league manager, ratings, replays y evaluación.
```

El hot loop no dependerá de ROS, ZeroMQ, Docker, Ray ni diccionarios Python. La ejecución más rápida será una llamada directa y batch desde Python hacia un núcleo Rust. Las tecnologías distribuidas se utilizarán únicamente cuando exista una frontera real entre procesos o máquinas.

---

## 2. Contexto y restricciones fijas

La infraestructura de desarrollo ya está terminada y no forma parte de este proyecto.

### 2.1 Devbox validada

```text
Host:
  Windows 11 10.0.26200.8875
  WSL 2.7.11.0
  Kernel 6.18.33.2-microsoft-standard-WSL2
  Ubuntu 26.04 LTS

Cómputo:
  NVIDIA GeForce RTX 3070
  8 GiB VRAM
  CUDA Driver API 13.1
  10 CPU lógicos visibles en WSL
  ~18 GiB RAM
  15 GiB swap

Contenedores:
  Docker Desktop 4.83.0
  Docker Engine 29.6.2
  backend WSL2
  único daemon

Validaciones:
  CUDA desde WSL
  CUDA desde Docker
  PyTorch 2.13.0+cu130
  multiprocessing
  DataLoader
  checkpoints
  reinicio completo
```

### 2.2 Reglas de infraestructura

El proyecto debe respetar estas decisiones:

- Windows es el único propietario del driver NVIDIA.
- No instalar drivers NVIDIA Linux en WSL.
- No instalar Docker Engine dentro de Ubuntu.
- No instalar NVIDIA Container Toolkit dentro de Ubuntu.
- No instalar globalmente CUDA Toolkit, PyTorch, OpenCV-CUDA, ROS 2 o Gazebo.
- Las dependencias del proyecto viven en imágenes OCI y archivos versionados.
- Los repositorios activos viven en `/home/rob/src`.
- Los builds viven en `/home/rob/work`.
- Los datos activos viven en `/home/rob/data`.
- Los runs viven en `/home/rob/runs`.
- Los checkpoints viven en `/home/rob/checkpoints`.
- Los replays viven en `/home/rob/replays`.
- No colocar cargas activas bajo `/mnt/c`, `/mnt/d` o `/mnt/g`.
- Todas las imágenes utilizadas para validación o releases deben fijarse por digest.

---

## 3. Problema

Los simuladores robóticos tradicionales ofrecen visualización, sensores y física general, pero suelen ser demasiado pesados para generar millones de transiciones RL. Los simuladores RL rápidos suelen estar acoplados a un motor, una API o un algoritmo específico. Las implementaciones multiagente frecuentemente introducen uno de estos problemas:

- una única política centralizada que genera las acciones de los tres robots;
- políticas independientes que especializan permanentemente cada identidad;
- roles explícitos que se quedan anclados a `robot_0`, `robot_1` o `robot_2`;
- observaciones con slots semánticos fijos que filtran la identidad;
- recompensas difíciles de auditar;
- imposibilidad de comparar física rápida con física de mayor fidelidad;
- entrenamiento y evaluación mezclados;
- ausencia de self-play histórico y pruebas fuera de distribución;
- protocolos incompatibles entre controladores;
- baja reproducibilidad de resultados;
- dificultad para que un agente de programación comprenda el flujo completo.

VSSS Lab debe permitir desarrollar una política colectiva en la que cada robot tome su propia decisión, todos puedan compartir pesos y las funciones tácticas roten según el estado del partido.

---

## 4. Visión del producto

> Crear la plataforma abierta de referencia para experimentar con coordinación multiagente, self-play y sim-to-real en VSSS, optimizada para iteración rápida, competencia reproducible y reemplazo independiente de cada componente.

La plataforma debe permitir responder preguntas como:

- ¿Una política compartida con crítico centralizado supera una estrategia heurística dinámica?
- ¿Los roles emergen sin introducir IDs de robot?
- ¿Qué arquitectura conserva mejor la simetría de permutación?
- ¿Cuánto rendimiento se obtiene con Rust CPU frente a un backend GPU?
- ¿Cuánta fidelidad física es necesaria antes de que el resultado del entrenamiento cambie?
- ¿Una política entrenada en el backend rápido conserva comportamiento en Gazebo?
- ¿Qué formas de domain randomization mejoran robustez?
- ¿Qué oponentes deben formar la liga para reducir olvido y ciclos estratégicos?
- ¿Qué checkpoint es verdaderamente mejor y no solo explota al último rival?

---

## 5. Fuentes de inspiración y principios extraídos

### 5.1 `juliodltv/simulation_vsss`

**Qué aporta:**

- campo 3D;
- seis robots diferenciales;
- pelota y colisiones;
- cámara cenital;
- comandos de ruedas;
- escenarios de lanzamiento;
- geometría, assets y parámetros existentes.

**Qué se reutiliza:**

- especificación física inicial;
- dimensiones y geometrías;
- assets permitidos por licencia;
- pruebas de referencia;
- escenarios para calibración;
- comportamiento observable como “golden model” inicial.

**Qué no se hereda como arquitectura principal:**

- ROS 1 Noetic;
- Catkin;
- Gazebo Classic;
- dependencia de topics por rueda en el hot loop;
- simulación 3D como generador masivo de experiencia.

### 5.2 RocketSim

**Principio extraído:** un motor especializado, standalone y headless puede priorizar throughput sobre generalidad y seguir siendo suficientemente fiel para entrenar agentes.

**Aplicación en VSSS Lab:**

- núcleo físico dedicado al dominio;
- timestep fijo;
- cero rendering en el hot loop;
- simulación batch;
- estructuras de memoria contiguas;
- feedback frecuente desde la política;
- benchmark por ticks por segundo y no por FPS visuales.

### 5.3 RLGym

**Principio extraído:** la dinámica física no debe definir observaciones, recompensas, acciones ni resets.

**Componentes equivalentes en VSSS Lab:**

```text
TransitionEngine  -> PhysicsBackend
StateMutator      -> ResetStrategy / StateMutator
ObsBuilder        -> ObservationBuilder
ActionParser      -> ActionAdapter
RewardFunction    -> RewardTerm / RewardPipeline
DoneCondition     -> TerminationCondition
Renderer          -> Renderer / ReplayViewer
```

Cada componente será reemplazable y testeable por separado.

### 5.4 rSoccer y VSSS-RL

**Principio extraído:** VSSS ya ha demostrado ser un dominio válido para RL, benchmarks de habilidades y sim-to-real.

**Aplicación:**

- revisar tareas y observaciones históricas;
- reproducir benchmarks comparables;
- establecer baselines;
- evitar redescubrir fallos conocidos;
- conservar compatibilidad conceptual sin copiar implementaciones obsoletas.

### 5.5 PettingZoo Parallel API

**Principio extraído:** VSSS es un juego parcialmente observable con acciones simultáneas.

**Aplicación:**

- API pública multiagente compatible con PettingZoo Parallel;
- adaptadores para herramientas externas;
- semántica explícita de observaciones, acciones, rewards, termination y truncation.

La representación interna no utilizará diccionarios Python en el hot path. El adaptador PettingZoo convertirá tensores batch a la interfaz pública.

### 5.6 TorchRL

**Principio extraído:** PyTorch-native, datos multiagente en tensores, parameter sharing y entrenamiento con actor descentralizado y crítico centralizado.

**Aplicación inicial:**

- IPPO como baseline de menor complejidad;
- MAPPO como baseline principal;
- política compartida por equipo;
- crítico centralizado durante entrenamiento;
- ejecución descentralizada;
- collectors y buffers auditables.

### 5.7 Molt

**Principio extraído:** el flujo completo debe ser suficientemente pequeño y explícito para que un investigador o un coding agent pueda modificarlo de extremo a extremo.

**Aplicación:**

- pocos procesos conceptuales;
- contrato canónico de trayectoria;
- policy version en cada rollout;
- colas asíncronas solamente cuando aporten throughput medido;
- losses y advantage estimators como funciones legibles;
- trainer separado del environment;
- evitar capas genéricas innecesarias.

Molt no se utilizará directamente porque su contrato es token-first y vLLM-first.

### 5.8 ROS 2 y Gazebo

**Principio extraído:** ROS/Gazebo son valiosos para integración robótica, sensores y validación, no para generar todos los rollouts.

**Aplicación:**

- backend separado;
- ROS 2 Lyrical sobre Ubuntu 26.04;
- Gazebo Jetty;
- `ros_gz`;
- bridge explícito desde el estado/acción canónicos;
- pruebas de equivalencia con el motor rápido.

---

## 6. Principios de diseño

1. **Contratos estables; implementaciones reemplazables.**
2. **Correctness antes que throughput.**
3. **Determinismo antes que distribución.**
4. **Batch interno; compatibilidad externa.**
5. **Una identidad física no implica una política distinta.**
6. **Los roles pertenecen al estado, no al ID.**
7. **Entrenamiento y evaluación son sistemas separados.**
8. **ROS no entra al hot loop.**
9. **ZeroMQ no entra al hot loop local.**
10. **La GPU se utiliza donde el perfil demuestre beneficio.**
11. **Toda optimización requiere benchmark reproducible.**
12. **Toda política promovida debe superar evaluación contra una liga.**
13. **Toda dependencia se fija por versión, commit o digest.**
14. **La política no conoce el backend físico.**
15. **El repositorio debe ser legible de extremo a extremo por agentes de desarrollo.**

---

## 7. Objetivos

### 7.1 Objetivos funcionales

- Crear un motor VSSS headless y determinista.
- Simular partidos 1v0, 1v1, 3v0 y 3v3.
- Exponer estados y acciones canónicas.
- Soportar comandos de velocidad de ruedas y velocidad lineal/angular.
- Crear observaciones globales y agent-centric.
- Implementar política compartida por los tres robots.
- Implementar IPPO y MAPPO.
- Implementar controladores heurísticos.
- Implementar self-play y liga histórica.
- Implementar match server externo.
- Registrar replays completos.
- Implementar backend de validación ROS 2/Gazebo.
- Ejecutar entrenamiento y evaluación en contenedores reproducibles.
- Soportar CPU en macOS para desarrollo, tests y controladores.
- Soportar CUDA en la devbox para learners e inferencia.

### 7.2 Objetivos de investigación

- Medir especialización por identidad.
- Medir rotación de funciones.
- Comparar roles emergentes, explícitos y latentes.
- Comparar IPPO vs MAPPO.
- Comparar MLP, Deep Sets, atención y GNN.
- Comparar actor feed-forward vs recurrente.
- Comparar recompensas densas y sparse.
- Evaluar domain randomization.
- Evaluar currículos y self-play.
- Evaluar sim-to-sim y sim-to-real.

### 7.3 Objetivos de operación

- Un comando para levantar el entorno.
- Un comando para entrenar.
- Un comando para evaluar.
- Un comando para ejecutar un torneo.
- Un comando para reproducir un partido.
- Un comando para ejecutar doctors.
- Un comando para crear un reporte de benchmark.

---

## 8. No objetivos de la primera versión

- No entrenar desde píxeles en el primer milestone.
- No implementar visión completa antes de validar estado estructurado.
- No desplegar Kubernetes.
- No construir un backend GPU antes de medir el backend CPU.
- No reemplazar inmediatamente Rapier por un solver físico propio.
- No controlar robots físicos en el primer release.
- No crear una UI web compleja.
- No usar LLMs dentro de la política de control.
- No garantizar determinismo bit a bit entre arquitecturas diferentes.
- No hacer ROS 2 una dependencia del paquete central.
- No utilizar una única red que produzca conjuntamente las seis acciones como baseline principal.

---

## 9. Usuarios y casos de uso

### 9.1 Investigador/desarrollador

- crea una nueva recompensa;
- cambia una observación;
- implementa una arquitectura;
- ejecuta seeds;
- compara resultados;
- inspecciona replays;
- calibra física.

### 9.2 Competidor

- implementa un controller;
- empaca el controller;
- lo conecta al match server;
- ejecuta partidos reproducibles;
- compara ratings;
- comparte replays.

### 9.3 Agente de desarrollo

- lee un issue con contrato y acceptance criteria;
- modifica un módulo aislado;
- ejecuta tests y benchmarks;
- abre un PR;
- produce evidencias;
- no necesita entender ROS para modificar rewards;
- no necesita entender RL para corregir física.

### 9.4 Operador

- inicia entrenamiento;
- desconecta SSH;
- consulta métricas;
- recupera un checkpoint;
- cambia entre Training y Gaming;
- ejecuta un torneo después de un entrenamiento.

---

## 10. Arquitectura de alto nivel

```text
                         ┌──────────────────────────┐
                         │      League Manager      │
                         │ matchmaking / ratings    │
                         │ promotion / evaluation   │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │         Learner          │
                         │ IPPO / MAPPO / future    │
                         └────────────┬─────────────┘
                                      │ trajectories
                         ┌────────────▼─────────────┐
                         │      Environment API     │
                         │ obs/action/reward/reset  │
                         └────────────┬─────────────┘
                                      │ canonical state
        ┌─────────────────────────────┼──────────────────────────────┐
        │                             │                              │
┌───────▼────────┐          ┌─────────▼────────┐          ┌──────────▼────────┐
│ Rust CPU batch │          │ Experimental GPU │          │ ROS 2 / Gazebo    │
│ primary engine │          │ MJX/other later  │          │ validation backend│
└───────┬────────┘          └──────────────────┘          └──────────┬────────┘
        │                                                            │
        └──────────────► Replay / Golden Tests ◄─────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │       Match Server       │
                         │ external controllers     │
                         │ ZeroMQ + FlatBuffers     │
                         └──────────────────────────┘
```

---

## 11. Decisiones tecnológicas iniciales

| Capa | Decisión |
|---|---|
| Monorepo | Sí |
| Núcleo físico | Rust 2024 |
| Física MVP | Rapier2D detrás de adapter |
| Bindings Python | PyO3 + maturin |
| Python | versión fijada por imagen; objetivo 3.13 |
| Gestión Python | uv |
| Build/commands | just |
| Learner | PyTorch + TorchRL |
| API pública MARL | PettingZoo Parallel |
| API single-agent | Gymnasium adapters |
| API interna | tensores/arrays batch |
| Algoritmos iniciales | IPPO y MAPPO |
| Política | shared parameters por equipo |
| Crítico | centralizado para MAPPO |
| IPC local | llamada directa; después shared memory |
| Protocolo remoto | ZeroMQ + FlatBuffers |
| Replay | FlatBuffers versionado |
| Orquestación inicial | procesos locales + Docker Compose |
| Distribución futura | Ray, solo después de benchmark |
| Robótica | ROS 2 Lyrical |
| Validación 3D | Gazebo Jetty |
| Tracking | MLflow local inicialmente |
| Métricas | TensorBoard-compatible + JSONL/Parquet |
| Tests | pytest, cargo test, property tests, golden tests |
| CI | GitHub Actions CPU; GPU self-hosted opcional y protegido |

### 11.1 Por qué Rapier2D primero

Rapier2D permite crear un motor 2D rápido, con SIMD y determinismo local bajo condiciones controladas. Se utilizará como implementación de referencia detrás de una interfaz propia. Si el profiling demuestra que su solver o estructuras limitan el throughput, podrá sustituirse por un kernel especializado sin cambiar el resto del sistema.

### 11.2 Por qué Rust

- control de memoria;
- seguridad en paralelización;
- estructuras batch eficientes;
- bindings Python mantenibles;
- fácil distribución de wheels;
- posibilidad de CLI y servidor nativos;
- menor riesgo que C++ para cambios frecuentes realizados por agentes.

---

## 12. Especificación canónica

El paquete `vsss-spec` no dependerá de Rapier, PyTorch, ROS o Python.

### 12.1 Unidades

- distancia: metros;
- tiempo: segundos;
- ángulos: radianes;
- velocidad lineal: m/s;
- velocidad angular: rad/s;
- velocidad de rueda: rad/s;
- masa: kg;
- fuerzas: N;
- torques: N·m.

No se permiten unidades implícitas.

### 12.2 Sistema de coordenadas

Definir un sistema de coordenadas único:

```text
origen: centro del campo
+x: hacia la portería amarilla en canonical orientation
+y: lateral izquierdo visto desde el equipo azul
theta=0: orientado hacia +x
rotación positiva: antihoraria
```

Los adaptadores pueden reflejar el campo para que cada política observe siempre que ataca hacia `+x`.

### 12.3 Entidades

```rust
struct Pose2 {
    x: f32,
    y: f32,
    theta: f32,
}

struct Twist2 {
    vx: f32,
    vy: f32,
    omega: f32,
}

struct RobotState {
    id: RobotId,
    team: Team,
    pose: Pose2,
    twist: Twist2,
    wheel_speed_left: f32,
    wheel_speed_right: f32,
    enabled: bool,
}

struct BallState {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    omega: f32,
}

struct MatchState {
    schema_version: u32,
    tick: u64,
    simulation_time: f32,
    score_blue: u16,
    score_yellow: u16,
    ball: BallState,
    robots: [RobotState; 6],
    events: EventFlags,
}
```

### 12.4 Acciones

```rust
enum ControlMode {
    WheelVelocity,
    BodyVelocity,
}

struct RobotAction {
    mode: ControlMode,
    left: f32,
    right: f32,
}
```

Para `BodyVelocity`, los campos representan `v` y `omega` y se convierten mediante un `ActionAdapter`.

### 12.5 Configuración de partido

Debe incluir:

- field geometry;
- robot geometry;
- ball properties;
- wheel geometry;
- timestep;
- control frequency;
- actuator limits;
- friction;
- restitution;
- match duration;
- reset rules;
- randomization ranges;
- seed;
- backend settings.

La configuración completa se serializa con cada run.

---

## 13. Motor físico rápido

### 13.1 Requisitos

- headless;
- fixed timestep;
- estado determinista en la misma plataforma y versión;
- batch de mundos;
- reset individual por mundo;
- seis robots y una pelota;
- differential drive;
- saturación de actuadores;
- aceleración y retardo configurables;
- colisiones robot-robot, robot-ball, ball-wall y robot-wall;
- detección de gol;
- eventos;
- sin asignaciones Python por tick;
- sin logging por tick en modo benchmark;
- snapshot y restore;
- checksum de estado.

### 13.2 Diseño batch

Representación preferida:

```text
position_x[world, entity]
position_y[world, entity]
angle[world, robot]
velocity_x[world, entity]
velocity_y[world, entity]
angular_velocity[world, entity]
wheel_action[world, robot, 2]
```

La API Python debe aceptar y devolver arrays contiguos:

```python
next_state, events = engine.step(actions)
```

Shapes:

```text
actions: [num_worlds, 6, 2]
state:   [num_worlds, state_dim]
events:  [num_worlds, event_dim]
```

### 13.3 Reference mode y optimized mode

**Reference mode:**

- checks adicionales;
- validaciones;
- trazas;
- útil para debugging.

**Optimized mode:**

- checks reducidos;
- logging deshabilitado;
- batch;
- SIMD/parallelism;
- útil para rollouts.

Los dos modos deben producir resultados equivalentes dentro de tolerancias definidas.

---

## 14. Environment layer

La composición se inspira en RLGym.

```python
env = VSSSEnv(
    backend=RustCpuBackend(...),
    reset_strategy=RandomizedKickoff(...),
    observation_builder=AgentCentricSetObservation(...),
    action_adapter=WheelVelocityAction(...),
    reward_pipeline=TeamRewardPipeline(...),
    termination_conditions=[
        GoalScored(),
        MatchTimeout(),
    ],
)
```

### 14.1 Contratos

```python
class PhysicsBackend(Protocol):
    def reset(self, config, seeds, initial_states=None): ...
    def step(self, actions): ...
    def snapshot(self): ...
    def restore(self, snapshots): ...

class ObservationBuilder(Protocol):
    def build(self, states, controlled_agents): ...

class ActionAdapter(Protocol):
    def parse(self, policy_actions, states): ...

class RewardTerm(Protocol):
    def compute(self, previous, actions, current, events): ...

class ResetStrategy(Protocol):
    def sample(self, seeds, config): ...

class TerminationCondition(Protocol):
    def evaluate(self, states, events): ...
```

### 14.2 Adaptadores

- PettingZoo Parallel;
- Gymnasium single-robot;
- Gymnasium centralized-team;
- TorchRL native;
- replay environment;
- remote match environment.

---

## 15. Observaciones

### 15.1 Observación agent-centric inicial

Cada robot recibe:

```text
self:
  orientación reflejada
  velocidad lineal local
  velocidad angular
  velocidad de ruedas

ball:
  posición relativa
  velocidad relativa
  distancia
  ángulo

goals:
  portería propia relativa
  portería rival relativa

teammates:
  conjunto de 2 embeddings relativos

opponents:
  conjunto de 3 embeddings relativos

context:
  tiempo restante
  marcador normalizado
  último evento
```

### 15.2 Prohibiciones iniciales

- No incluir ID absoluto del robot.
- No incluir `robot_0_is_striker`.
- No ordenar compañeros por identidad.
- No asociar rewards a un ID fijo.
- No permitir que el lado del campo cambie la semántica sin reflexión.

### 15.3 Tratamiento permutation-invariant/equivariant

Implementar progresivamente:

1. orden por distancia como baseline;
2. random permutation durante entrenamiento;
3. Deep Sets;
4. attention sobre conjuntos;
5. GNN opcional.

### 15.4 Memoria

Primera versión feed-forward. Segunda variante con GRU:

```text
hidden state separado por:
  world
  agent
  policy version
```

El reset de memoria debe ser explícito por agente y episodio.

---

## 16. Acciones

### 16.1 Acción principal

```text
[-1, 1] x [-1, 1]
```

para rueda izquierda y derecha, posteriormente escalada a límites físicos.

### 16.2 Variantes

- velocidad de ruedas;
- velocidad lineal/angular;
- target point + speed;
- acciones tácticas discretas para experimentos jerárquicos.

### 16.3 Seguridad y suavizado

- clipping;
- rate limits;
- lag configurable;
- dropped commands;
- motor asymmetry;
- dead zone;
- action repeat;
- penalización de saturación opcional.

---

## 17. Recompensas

La recompensa se compondrá de términos versionados:

```text
reward_total =
    goal_reward
  + concede_penalty
  + ball_progress
  + possession_proxy
  + defensive_positioning
  + spacing
  + action_smoothness
  + collision_penalty
  + optional_role_consistency
```

### 17.1 Reglas

- La recompensa de gol domina el objetivo.
- Los términos densos no deben permitir optimizar sin intentar anotar.
- Toda recompensa debe tener test unitario.
- Todo término debe poder deshabilitarse.
- La contribución de cada término debe registrarse por separado.
- Las recompensas de equipo son el baseline.
- Rewards individuales requieren justificación experimental.

### 17.2 Reward hacking tests

- girar sin avanzar;
- acorralar la pelota sin tirar;
- chocar para acumular progreso;
- oscilar cerca de un threshold;
- mantener posesión perdiendo el partido;
- explotar resets;
- provocar autogoles del adversario de manera no generalizable.

---

## 18. Multiagente y roles dinámicos

### 18.1 Baseline principal

```text
obs_blue_0 ─┐
obs_blue_1 ─┼── shared actor π_blue ──► acciones independientes
obs_blue_2 ─┘

global/team state ──► centralized critic V_blue
```

El equipo rival puede usar la misma política, una política histórica o un controller distinto.

### 18.2 Qué significa “verdaderamente multiagente”

- tres instancias de decisión;
- tres observaciones;
- tres estados recurrentes cuando aplique;
- tres acciones;
- parámetros compartidos opcionales;
- entrenamiento centralizado permitido;
- ejecución descentralizada.

### 18.3 Roles

Se soportarán tres experimentos:

1. **Emergentes:** no hay variable de rol.
2. **Asignación dinámica:** un módulo asigna objetivos según costes.
3. **Latentes:** la política aprende un embedding de intención.

Ninguno puede anclar permanentemente un rol a un ID.

### 18.4 Métricas de roles

- tasa de cambio de función;
- tiempo medio de compromiso;
- número de swaps por minuto;
- identidad que ejecuta cada función;
- entropía de asignación por identidad;
- distancia al balón al asumir presión;
- cobertura de portería;
- spacing entre compañeros;
- chattering;
- simetría bajo permutación.

### 18.5 Test bloqueante de identidad

Dado un estado físico, permutar únicamente los IDs de los robots y reconstruir la observación.

**Criterio:** las acciones deben permutarse de forma equivalente dentro de tolerancia. Si el comportamiento táctico cambia por renombrar agentes, el test falla.

---

## 19. Learner

### 19.1 Algoritmos iniciales

1. PPO single-agent para habilidades.
2. IPPO shared-parameter.
3. MAPPO shared-actor + centralized critic.
4. SAC/TD3 para tareas individuales como comparación off-policy.
5. Algoritmos adicionales solamente después de baselines reproducibles.

### 19.2 Trayectoria canónica

```text
run_id
episode_id
world_id
tick
agent_id
team
policy_id
policy_version
observation
action
log_prob
reward_total
reward_components
terminated
truncated
value
global_state_ref
```

### 19.3 Policy versioning

Todo rollout debe registrar la versión de pesos que lo generó.

En el modo síncrono inicial:

```text
collect rollout N
optimize policy N
publish policy N+1
```

En un modo asíncrono futuro:

- limitar staleness;
- rechazar rollouts demasiado antiguos;
- registrar lag;
- comparar estabilidad contra modo síncrono;
- no asumir que async es mejor sin benchmark.

### 19.4 Checkpoints

Guardar:

- actor;
- critic;
- optimizer;
- scaler;
- scheduler;
- normalization statistics;
- curriculum state;
- league state;
- RNG Python;
- RNG NumPy;
- RNG PyTorch CPU/GPU;
- environment seeds;
- git commit;
- container digest;
- configuration;
- metrics summary.

---

## 20. Curriculum

Orden inicial:

```text
C0: mover un robot a un target
C1: orientar y alcanzar pelota
C2: empujar pelota hacia dirección
C3: anotar sin adversario
C4: defender tiro simple
C5: 1v1
C6: 2v1
C7: 3v0 coordinación
C8: 3v3 contra heurística
C9: 3v3 self-play
C10: domain randomization
```

La promoción entre etapas utiliza métricas, no un número fijo de steps.

Ejemplo:

```text
promover cuando:
  success_rate >= threshold
  en N seeds de evaluación
  durante K evaluaciones consecutivas
```

Los valores concretos se definen por tarea.

---

## 21. Controladores heurísticos

Antes de RL deben existir:

- random legal;
- go-to-target;
- go-to-ball;
- shoot-to-goal;
- goalie;
- dynamic assignment;
- collision-aware team;
- scripted adversary configurable.

### 21.1 Dynamic assignment baseline

Crear una matriz de costes:

```text
             pressure ball   support   defend
robot A           0.2          0.8       0.9
robot B           0.7          0.3       0.6
robot C           0.9          0.5       0.1
```

Resolver asignación y aplicar:

- histéresis;
- coste de cambio;
- tiempo mínimo de compromiso;
- fallback si un robot está bloqueado.

Este baseline será rival, oracle parcial y referencia de coordinación.

---

## 22. League manager y self-play

### 22.1 Población inicial

- current main;
- checkpoints históricos;
- heurísticas;
- random policies;
- aggressive exploiter;
- defensive exploiter;
- friend policies;
- ablation policies.

### 22.2 Matchmaking inicial

Distribución configurable, por ejemplo:

```text
35% current main
25% historical
15% heuristic
15% exploiters
10% randomized
```

Los porcentajes son defaults experimentales, no constantes del sistema.

### 22.3 Promoción de checkpoint

Un candidato se promueve únicamente si:

- supera al current main;
- no regresa contra históricos;
- supera heurísticas mínimas;
- no aumenta autogoles o colisiones por encima del límite;
- pasa permutation tests;
- pasa seeds no vistas;
- pasa escenarios de latencia y ruido;
- cumple criterio estadístico predefinido.

### 22.4 Ratings

Implementar inicialmente Elo y conservar datos suficientes para añadir TrueSkill u otro sistema.

Registrar:

- win/draw/loss;
- goal difference;
- side;
- seed;
- opponent;
- policy versions;
- simulation config;
- duration;
- timeout;
- infra failures.

---

## 23. Match server

### 23.1 Propósito

Permitir que dos controladores compitan sin compartir implementación, lenguaje o framework.

### 23.2 Características

- servidor autoritativo;
- frecuencia de control fija;
- deadline por tick;
- acciones tardías descartadas o repetidas según regla;
- límites de comandos;
- side switching;
- seeds;
- logs;
- replay;
- aislamiento por proceso o contenedor;
- health checks;
- protocolo versionado.

### 23.3 Transporte

```text
mismo proceso:
  llamada directa

procesos locales:
  shared memory / Unix sockets

competencia y controllers remotos:
  ZeroMQ + FlatBuffers
```

No se utilizará ZeroMQ para cada step entre learner y backend dentro del mismo proceso.

### 23.4 Protocolo

Mensajes mínimos:

```text
Hello
Capabilities
MatchConfig
Reset
Observation
Action
Heartbeat
MatchEvent
MatchResult
Error
```

Cada mensaje incluye:

```text
protocol_version
match_id
sequence
timestamp
deadline
payload
```

FlatBuffers debe evolucionar añadiendo campos, sin renombrar ni reutilizar IDs de campos existentes.

---

## 24. Replay y observabilidad

### 24.1 Replay

Un replay debe permitir:

- reproducir visualmente;
- recalcular métricas;
- comparar políticas;
- detectar divergencia;
- convertir a dataset offline;
- usar como regression test.

Contenido:

- initial config;
- seeds;
- estados;
- acciones;
- events;
- rewards;
- policy versions;
- metadata de build;
- checksums.

### 24.2 Logs

Formato:

- logs humanos estructurados;
- JSONL para eventos;
- Parquet para análisis de episodios;
- TensorBoard/MLflow para curvas;
- no registrar cada estado en stdout.

### 24.3 Métricas de sistema

- ticks/s;
- agent steps/s;
- env reset/s;
- rollout latency;
- inference latency;
- learner update time;
- queue depth;
- staleness;
- CPU;
- RAM;
- VRAM;
- disk throughput;
- dropped actions;
- simulation divergence.

---

## 25. Backend ROS 2/Gazebo

### 25.1 Objetivo

Validar:

- geometría;
- dinámica diferencial;
- fricción;
- contactos;
- cámara;
- latencia;
- bridges;
- futura conexión a hardware.

### 25.2 Stack

```text
Ubuntu 26.04 container
ROS 2 Lyrical
Gazebo Jetty
ros_gz
ros2_control cuando sea necesario
```

### 25.3 Reutilización del simulador existente

Extraer:

- URDF/SDF;
- meshes;
- field;
- ball;
- camera placement;
- wheel geometry;
- physical parameters;
- launch scenarios.

No ejecutar el paquete ROS 1 como dependencia directa del proyecto principal.

### 25.4 Golden calibration scenarios

- aceleración recta;
- frenado;
- giro sobre el centro;
- giro con radios definidos;
- impacto frontal con pelota;
- impacto oblicuo;
- pelota contra pared;
- robot contra pared;
- colisión de dos robots;
- compresión de pelota;
- triple collision;
- pérdida de velocidad por fricción.

Comparar:

```text
position error
velocity error
heading error
time-to-stop
rebound angle
rebound speed
```

El backend rápido no tiene que coincidir perfectamente; debe mantenerse dentro de tolerancias documentadas para los fenómenos relevantes.

---

## 26. Visión y OpenCV-CUDA

La visión queda fuera del MVP, pero la arquitectura debe permitirla.

### 26.1 Variantes

- `vsss-vision-cpu` para macOS y CI;
- `vsss-vision-cuda` para RTX 3070;
- builder/runtime multistage;
- compute capability 8.6;
- OpenCV contrib;
- headless;
- GStreamer opcional;
- NVDEC/NVENC opcional.

### 26.2 Regla de adopción

OpenCV-CUDA solo se integra después de demostrar que:

- existe una pipeline visual definida;
- CPU es cuello de botella;
- la transferencia y sincronización no eliminan el beneficio;
- el contenedor pasa `cv2.cuda` smoke tests;
- la variante CPU sigue funcionando.

---

## 27. Contenedores

### 27.1 Matriz inicial

```text
vsss-dev-cpu
  Python, uv, Rust, maturin, test tools

vsss-train-cuda
  PyTorch, TorchRL, CUDA runtime

vsss-sim-build
  Rust toolchain, native build dependencies

vsss-ros
  Ubuntu 26.04, ROS 2 Lyrical, Gazebo Jetty

vsss-vision-cpu
  OpenCV headless

vsss-vision-cuda
  OpenCV CUDA, opcional
```

### 27.2 Reglas

- imágenes fijadas por digest;
- non-root user;
- cache BuildKit;
- wheels construidos una vez;
- runtime sin compiladores cuando sea razonable;
- health checks;
- labels OCI;
- SBOM;
- no incluir secretos;
- no montar Docker socket en el contenedor de entrenamiento;
- bind mounts explícitos para runs y checkpoints.

### 27.3 MacBooks

Los desarrolladores en macOS deben poder:

- ejecutar tests de spec;
- compilar Rust;
- ejecutar backend CPU;
- ejecutar controladores;
- reproducir replays;
- correr episodios pequeños;
- no requerir CUDA ni ROS.

---

## 28. Estructura del repositorio

```text
vsss-lab/
├── README.md
├── LICENSE
├── NOTICE
├── CONTRIBUTING.md
├── SECURITY.md
├── AGENTS.md
├── justfile
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── Cargo.toml
├── Cargo.lock
│
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── protocols/
│   ├── experiments/
│   └── calibration/
│
├── schemas/
│   ├── match.fbs
│   ├── replay.fbs
│   └── trajectory.fbs
│
├── crates/
│   ├── vsss-spec/
│   ├── vsss-physics-api/
│   ├── vsss-physics-rapier/
│   ├── vsss-batch/
│   ├── vsss-replay/
│   ├── vsss-protocol/
│   ├── vsss-match-server/
│   └── vsss-python/
│
├── python/
│   ├── vsss_env/
│   ├── vsss_observations/
│   ├── vsss_actions/
│   ├── vsss_rewards/
│   ├── vsss_baselines/
│   ├── vsss_train/
│   ├── vsss_league/
│   ├── vsss_eval/
│   ├── vsss_analysis/
│   └── vsss_cli/
│
├── backends/
│   ├── ros_gazebo/
│   ├── replay/
│   └── experimental/
│
├── containers/
│   ├── dev-cpu/
│   ├── train-cuda/
│   ├── ros/
│   └── vision/
│
├── benchmarks/
│   ├── physics/
│   ├── bindings/
│   ├── inference/
│   ├── ipc/
│   └── end_to_end/
│
├── experiments/
│   ├── configs/
│   └── reports/
│
├── tests/
│   ├── golden/
│   ├── integration/
│   ├── property/
│   └── performance/
│
└── tools/
    ├── doctors/
    ├── migration/
    ├── replay_viewer/
    └── profiling/
```

---

## 29. CLI y experiencia de desarrollo

Comandos objetivo:

```bash
just doctor
just bootstrap
just build
just test
just lint
just benchmark
just benchmark-physics
just sim
just play
just train CONFIG=...
just evaluate CHECKPOINT=...
just tournament
just replay FILE=...
just ros-up
just ros-calibrate
just report RUN=...
```

Cada comando:

- imprime configuración efectiva;
- falla temprano;
- guarda evidencia;
- utiliza rutas estándar;
- no requiere conocimiento de comandos internos.

---

## 30. Configuración

Usar archivos declarativos versionados.

```yaml
experiment:
  name: mappo_shared_set_obs_v1
  seed: 1234

environment:
  backend: rust_rapier
  num_worlds: 256
  control_hz: 40
  physics_hz: 120
  task: vsss_3v3

policy:
  sharing: team
  architecture: deepsets_mlp
  recurrent: false

algorithm:
  name: mappo
  rollout_steps: 256
  gamma: 0.99
  gae_lambda: 0.95

league:
  enabled: true
  historical_fraction: 0.25
```

La configuración efectiva completa se copia al run.

No se permiten defaults invisibles que afecten resultados.

---

## 31. Benchmarks y SLOs

Los targets iniciales son provisionales y se revisan después del primer benchmark de la devbox.

### 31.1 Correctness gates

- cero NaN/Inf en 10 millones de ticks aleatorios;
- determinismo local en replay con checksum idéntico;
- snapshot/restore exacto;
- no penetración sostenida mayor al límite;
- detección de gol consistente;
- tests de unidad para reglas;
- property tests para simetrías.

### 31.2 Throughput targets iniciales

En la RTX 3070 devbox, backend CPU headless:

- **single-thread:** objetivo ≥10,000 world ticks/s en escenario 3v3;
- **10 CPU lógicos:** objetivo ≥75,000 world ticks/s agregado;
- binding Python: overhead ≤15% frente al benchmark Rust puro;
- reset batch: ≥10,000 resets/s;
- memoria: sin crecimiento no explicado durante 1 hora de stress;
- p99 step latency registrada.

Estos valores no bloquean el MVP si correctness no está listo. Sí bloquean declarar optimización final.

### 31.3 RL targets iniciales

- skill go-to-target alcanza ≥95% success en seeds de evaluación;
- skill score-empty-goal supera heurística random;
- 1v1 RL supera random y compite con scripted baseline;
- 3v3 policy pasa permutation test;
- policy compartida utiliza las tres identidades en funciones ofensivas/defensivas;
- candidato promovido supera current main bajo evaluación predefinida.

### 31.4 Reproducibilidad

- misma seed y build produce replay checksum idéntico en misma plataforma;
- run restaurado continúa sin pérdida de optimizer/league state;
- cada resultado tiene git commit y digest de imagen;
- cada benchmark genera JSON machine-readable.

---

## 32. Testing

### 32.1 Rust

- unit tests;
- proptest;
- cargo clippy;
- cargo fmt;
- sanitizer jobs cuando sea posible;
- benchmarks Criterion;
- fuzzing para parsers/protocolos.

### 32.2 Python

- pytest;
- mypy o pyright;
- ruff;
- coverage;
- hypothesis;
- integration tests;
- deterministic seeds.

### 32.3 Contract tests

Cada backend debe pasar el mismo suite:

```text
reset
step
snapshot
restore
goal detection
limits
reflection
permutation
serialization
replay
```

### 32.4 Performance regression

Guardar baseline con:

- hardware fingerprint;
- OS/container;
- compiler;
- commit;
- benchmark config.

Fallar CI solo ante regresiones grandes y reproducibles. Las pruebas GPU se ejecutan en runner protegido, nunca para código no confiable de forks.

---

## 33. Seguridad y supply chain

- Dependabot/Renovate con PRs, no auto-merge indiscriminado.
- Digests OCI.
- Cargo.lock y uv.lock.
- SBOM para imágenes.
- secret scanning.
- ninguna credencial en configs.
- controllers externos sin acceso al Docker socket.
- match server aplica timeouts y límites de recursos.
- self-hosted runner no ejecuta PRs no confiables.
- third-party assets registrados en `NOTICE`.
- no reutilizar código de Rocket League ni assets propietarios.

---

## 34. Flujo de agentes de desarrollo

Cada issue delegable debe incluir:

```text
Context
Problem
Scope
Out of scope
Interfaces allowed to change
Acceptance criteria
Tests required
Benchmark required
Artifacts/evidence
Rollback
```

El agente debe:

1. leer `AGENTS.md`;
2. ejecutar `just doctor`;
3. crear branch;
4. implementar el cambio mínimo coherente;
5. ejecutar tests;
6. ejecutar benchmark cuando aplique;
7. actualizar ADR/documentación;
8. abrir PR con evidencias;
9. no modificar contratos sin ADR;
10. no promover resultados RL sin evaluación.

### 34.1 ADRs

Crear Architecture Decision Records para:

- canonical coordinates;
- physics backend;
- units;
- replay schema;
- shared policy semantics;
- IPC;
- container bases;
- experiment tracking;
- ROS bridge.

---

## 35. Roadmap por milestones

## M0 — Bootstrap del monorepo

**Entregables:**

- repositorio;
- licencia;
- estructura;
- justfile;
- containers dev CPU y train CUDA;
- CI CPU;
- doctors;
- AGENTS.md;
- ADR template;
- manifest de plataforma.

**Gate:**

```text
just doctor
just build
just test
```

funcionan en WSL y Mac CPU.

## M1 — Especificación canónica

**Entregables:**

- units;
- geometry;
- entities;
- actions;
- events;
- config;
- reflection;
- serialization;
- golden fixtures.

**Gate:** tests de reglas y simetría.

## M2 — Physics backend MVP

**Entregables:**

- Rapier2D backend;
- differential drive;
- ball;
- walls;
- goals;
- snapshot/restore;
- batch básico;
- benchmark.

**Gate:** correctness suite y replay determinista.

## M3 — Python bindings y environment API

**Entregables:**

- PyO3/maturin;
- internal batch API;
- PettingZoo adapter;
- Gymnasium adapters;
- RLGym-like composition;
- random environment tests.

**Gate:** overhead medido y API contract tests.

## M4 — Heuristic baselines

**Entregables:**

- go-to-target;
- go-to-ball;
- goalie;
- dynamic assignment;
- match runner;
- replay viewer mínimo.

**Gate:** partido 3v3 scripted reproducible.

## M5 — RL skills

**Entregables:**

- PPO skill trainer;
- configs;
- checkpoint/resume;
- metrics;
- curriculum C0-C5.

**Gate:** habilidades alcanzan thresholds.

## M6 — MARL baseline

**Entregables:**

- IPPO;
- MAPPO;
- shared policy;
- centralized critic;
- permutation-invariant observation;
- identity tests;
- curriculum 3v3.

**Gate:** política multiagente pasa tests de identidad y supera random.

## M7 — League y self-play

**Entregables:**

- registry;
- matchmaking;
- evaluation workers;
- Elo;
- promotion;
- historical checkpoints;
- tournament reports.

**Gate:** promoción reproducible y no regresiva.

## M8 — Match server externo

**Entregables:**

- FlatBuffers;
- ZeroMQ;
- controller SDK Python/Rust;
- deadlines;
- containers;
- replays;
- torneo Roberto vs Julio.

**Gate:** controllers heterogéneos compiten sin compartir proceso.

## M9 — Calibración contra simulador de referencia

**Entregables:**

- extracción de parámetros/assets;
- container legacy si es necesario;
- golden scenarios;
- calibration report;
- tolerancias.

**Gate:** diferencias documentadas y aceptables.

## M10 — ROS 2/Gazebo backend

**Entregables:**

- container ROS Lyrical/Gazebo Jetty;
- migrated assets;
- canonical bridge;
- visual replay;
- sim-to-sim tests.

**Gate:** una misma política ejecuta en fast sim y Gazebo sin cambiar su API.

## M11 — Domain randomization y robustez

**Entregables:**

- friction;
- motor variation;
- latency;
- action drops;
- observation noise;
- OOD evaluation.

**Gate:** política robusta supera policy no randomizada en suite OOD.

## M12 — Visión y hardware

**Entregables:**

- camera pipeline;
- vision CPU/CUDA;
- estimator;
- ROS bridge;
- hardware-in-the-loop;
- safety gates.

**Gate:** definido posteriormente.

---

## 36. Secuencia propuesta de pull requests

1. `chore: bootstrap vsss-lab monorepo`
2. `feat(spec): add canonical units geometry and state`
3. `feat(protocol): add versioned FlatBuffers schemas`
4. `feat(physics): add Rapier2D reference backend`
5. `feat(batch): add vectorized world stepping`
6. `feat(python): add PyO3 bindings and wheels`
7. `feat(env): add composable environment API`
8. `feat(api): add PettingZoo and Gymnasium adapters`
9. `feat(baselines): add heuristic controllers`
10. `feat(replay): add deterministic replay and viewer`
11. `feat(train): add single-agent PPO skills`
12. `feat(marl): add IPPO shared-policy baseline`
13. `feat(marl): add MAPPO centralized critic`
14. `feat(eval): add identity and permutation evaluation`
15. `feat(league): add self-play registry and Elo`
16. `feat(match): add external match server`
17. `feat(ros): add Lyrical/Jetty validation backend`

Cada PR debe ser pequeño, ejecutable y evitar mezclar infraestructura, algoritmo y física salvo necesidad explícita.

---

## 37. Riesgos y mitigaciones

### Física rápida incorrecta

**Mitigación:** golden tests, backend de referencia, calibración Gazebo y tolerancias por fenómeno.

### Roles anclados a identidad

**Mitigación:** política compartida, no IDs, permutación, set encoders, métricas y test bloqueante.

### Reward hacking

**Mitigación:** replay review, componentes registrados, adversarial tests y sparse objective dominante.

### Sobreingeniería prematura

**Mitigación:** no Ray/GPU/custom solver antes de benchmarks.

### WSL y rendering

**Mitigación:** headless por defecto; backend ROS en contenedor; visualización remota opcional.

### VRAM limitada

**Mitigación:** simulación CPU, buffers en RAM, modelos pequeños, mixed precision, batch configurable.

### Dependencias latest inestables

**Mitigación:** explorar latest en branches, fijar versiones y digests por run, rollback reproducible.

### Agentes modifican contratos accidentalmente

**Mitigación:** ADR obligatorio, contract tests y CODEOWNERS en schemas/spec.

### Self-play colapsa

**Mitigación:** históricos, heurísticas, exploiters, diversidad de oponentes y evaluación fija.

### Diferencias Mac/WSL

**Mitigación:** core CPU portable, tolerancias numéricas, CI en ambas plataformas y releases de wheels.

---

## 38. Definition of Done del producto inicial

La primera versión utilizable se considera terminada cuando:

- existe monorepo reproducible;
- backend rápido simula 3v3;
- existe política heurística;
- existe IPPO y MAPPO;
- los tres robots toman decisiones separadas;
- la política puede compartir pesos;
- el test de permutación pasa;
- existe torneo contra históricos y heurísticas;
- los runs se pueden reanudar;
- los replays se pueden reproducir;
- el match server permite competir con controladores separados;
- el mismo contrato ejecuta sobre fast sim y Gazebo;
- toda la plataforma funciona desde contenedores sin instalar dependencias globales;
- un agente puede implementar un reward o backend siguiendo interfaces documentadas.

---

## 39. Primera tarea para el agente implementador

### Objetivo

Crear el monorepo `vsss-lab` y completar M0 sin comenzar todavía el motor físico.

### Alcance

- crear repositorio;
- añadir licencia, README, CONTRIBUTING, SECURITY y AGENTS;
- crear workspace Rust;
- crear workspace Python con uv;
- crear `justfile`;
- crear contenedor CPU;
- crear contenedor CUDA de smoke test;
- crear CI CPU;
- crear doctors;
- crear ADR template;
- crear platform manifest;
- crear estructura de directorios;
- añadir lint/test placeholders reales;
- documentar rutas persistentes de la devbox.

### Acceptance criteria

```bash
git clone ...
cd vsss-lab
just doctor
just bootstrap
just build
just test
```

deben funcionar desde:

- WSL2 Ubuntu 26.04;
- contenedor dev CPU;
- MacBook CPU para tests que no requieran CUDA.

Además:

- `just cuda-smoke` pasa en devbox-gpu;
- no se instala nada global;
- CI pasa;
- todos los containers corren como non-root;
- imágenes base están fijadas por digest;
- no existen secretos;
- `AGENTS.md` contiene reglas de contribución para agentes.

### Fuera de alcance

- Rapier;
- PyTorch training loop;
- ROS;
- Gazebo;
- ZeroMQ;
- FlatBuffers schemas definitivos;
- UI;
- self-play.

---

## 40. Referencias

1. `simulation_vsss`: https://github.com/juliodltv/simulation_vsss  
2. RocketSim: https://github.com/ZealanL/RocketSim  
3. RLGym overview: https://rlgym.org/Getting%20Started/overview/  
4. RLGym custom environments: https://rlgym.org/Custom%20Environments/custom-environment/  
5. PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/  
6. TorchRL Multi-Agent PPO tutorial: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html  
7. TorchRL MAPPO loss: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.multiagent.MAPPOLoss.html  
8. RLlib multi-agent environments: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html  
9. Molt paper: https://arxiv.org/html/2607.21653v1  
10. Molt repository: https://github.com/NVIDIA-NeMo/labs-molt  
11. rSoccer: https://github.com/robocin/rSoccer  
12. rSoccer paper: https://arxiv.org/abs/2106.12895  
13. VSSS-RL sim-to-real: https://arxiv.org/abs/2003.11102  
14. Rapier: https://rapier.rs/docs/  
15. Rapier determinism: https://rapier.rs/docs/user_guides/rust/determinism/  
16. PyO3: https://pyo3.rs/  
17. maturin: https://www.maturin.rs/  
18. FlatBuffers schema evolution: https://flatbuffers.dev/evolution/  
19. ZeroMQ Guide: https://zguide.zeromq.org/docs/  
20. ROS 2 Lyrical: https://docs.ros.org/en/lyrical/  
21. ROS/Gazebo compatibility: https://gazebosim.org/docs/latest/ros_installation/  
22. ROS/Gazebo bridge: https://gazebosim.org/docs/latest/ros2_integration/

---

## 41. Nota de decisión

Este PRD distingue entre:

- **decisiones cerradas:** contratos, separación de capas, política compartida, backend CPU inicial, contenedores, API y validación;
- **hipótesis a medir:** Rapier frente a solver propio, CPU frente a GPU, síncrono frente a asíncrono, Deep Sets frente a GNN y ZeroMQ frente a otra opción remota;
- **trabajo futuro:** visión, robots reales y simulación GPU.

Ninguna tecnología debe conservarse por prestigio o novedad. Se conserva cuando supera los contratos, tests y benchmarks del proyecto.
