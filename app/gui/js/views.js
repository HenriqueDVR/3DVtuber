// The 3D views (Parts, Joints, Face, Movement): one renderer each, orbit, and camera buttons.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Blender (Z up, facing -Y) <-> glTF/three (Y up, facing +Z)
export const toThree = ([x, y, z]) => new THREE.Vector3(x, z, -y);
export const toBlender = (v) => [+v.x.toFixed(4), +(-v.z).toFixed(4), +v.y.toFixed(4)];

// Where each camera button looks from; the character faces +Z, their left is +X.
const CAMS = { Front: [0, 0, 1], Side: [1, 0, 0], Back: [0, 0, -1] };

export function makeView(host) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0e12);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(1, 2, 2); scene.add(sun);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
  camera.position.set(0, 0.9, 3.2);
  // a press of the middle button starts the browser's own scrolling (the page slid up and down
  // while the view was being moved): the button belongs to the view
  renderer.domElement.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); });
  renderer.domElement.addEventListener("auxclick", (e) => e.preventDefault());
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0.85, 0); orbit.update();
  const view = { renderer, scene, camera, orbit, host, model: null,
    resize() {
      const w = host.clientWidth, h = host.clientHeight;
      if (!w || !h) return;  // on a hidden page: resized when it shows
      renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
      // a model loaded while its page was hidden is framed once the view has a size
      if (view.pendingLook && view.model) { view.look(view.pendingLook); view.pendingLook = null; }
    },
    // frames the model now, or when the view first shows
    frame(from = [0, 0, 1]) { view.pendingLook = from; view.resize(); },
    clear() { if (view.model) scene.remove(view.model); view.model = null; },
    pick(event, objects) {
      const r = renderer.domElement.getBoundingClientRect();
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1), camera);
      return ray.intersectObjects(objects.filter((o) => o.visible !== false), true)[0];
    },
    // the whole model in view, from `from` (a camera button) or from where the camera is now
    look(from) {
      if (!view.model) return;
      const box = new THREE.Box3().setFromObject(view.model), size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());
      const dir = from ? new THREE.Vector3(...from) : camera.position.clone().sub(orbit.target).normalize();
      const distance = (Math.max(size.y, size.x / camera.aspect) / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12 + size.z / 2;
      orbit.target.copy(centre);
      camera.position.copy(centre).addScaledVector(dir, distance);
      orbit.update();
    } };
  const cams = document.createElement("div");
  cams.className = "cams";
  cams.innerHTML = Object.keys(CAMS).map((k) => `<button data-cam="${k}">${k}</button>`).join("") + `<button data-cam="fit" title="the whole model in view">Fit</button>`;
  cams.onclick = (e) => { const k = e.target.dataset.cam; if (k) view.look(CAMS[k]); };
  host.append(cams);
  // drawn while its page shows. Not by offsetParent: that is null for the Paint view over the whole
  // window (position: fixed), which then stayed black
  renderer.setAnimationLoop(() => { if (host.clientWidth && host.clientHeight) renderer.render(scene, camera); });
  // a view on a hidden page has no size; it gets one when its page shows
  new ResizeObserver(() => view.resize()).observe(host);
  view.resize();
  return view;
}
