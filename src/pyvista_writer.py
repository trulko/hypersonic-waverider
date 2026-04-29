"""
PyVista (VTK) renderers for waverider surface meshes and flowfield slices.

Public API
----------
plot_scalar_field_pv(...)
plot_flowfield_slices_pv(...)
plot_geometry_views_pv(...)
"""

import numpy as np
from matplotlib import cm, colors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mesh_to_polydata(mesh):
    """Convert one of our mesh dicts to a pyvista PolyData triangle mesh.

    Each triangle gets its own three points — fine for cell-scalar
    visualisation, and avoids subtle joinery bugs across the symmetry plane.
    """
    import pyvista as pv
    tris = mesh["triangles"]
    n_tri = tris.shape[0]
    verts = tris.reshape(-1, 3).astype(float)
    idx   = np.arange(3 * n_tri, dtype=np.int64).reshape(n_tri, 3)
    faces = np.concatenate(
        [np.full((n_tri, 1), 3, dtype=np.int64), idx], axis=1
    ).ravel()
    return pv.PolyData(verts, faces)


def _camera_from_elev_azim(elev_deg, azim_deg, focal, distance,
                           view_up=(0.0, 0.0, 1.0)):
    """Build a PyVista ``camera_position`` triple from matplotlib-style
    ``(elev, azim)`` angles relative to ``focal``."""
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    off = distance * np.array([
        np.cos(e) * np.cos(a),
        np.cos(e) * np.sin(a),
        np.sin(e),
    ])
    pos = np.asarray(focal, dtype=float) + off
    return [tuple(pos), tuple(focal), tuple(view_up)]


def _quantize_rgb(values, norm, cmap_obj, n_colors):
    """Snap ``values`` onto the n_colors-cell midpoints so on-mesh colors
    line up exactly with the discrete scalar bar."""
    if norm is not None:
        t = np.clip(norm(np.asarray(values, dtype=float)), 0.0, 1.0)
    else:
        v = np.asarray(values, dtype=float)
        t = np.clip((v - v.min()) / max(v.max() - v.min(), 1e-30), 0.0, 1.0)
    t = (np.floor(t * n_colors).clip(0, n_colors - 1) + 0.5) / n_colors
    return (cmap_obj(t)[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Surface scalar field
# ---------------------------------------------------------------------------
def plot_scalar_field_pv(lower_mesh, lower_field,
                         upper_mesh=None, upper_field=None,
                         cmap="viridis",
                         colorbar_label="Value",
                         vmin=None, vmax=None,
                         lower_alpha=1.0, upper_alpha=1.0,
                         norm=None,
                         n_colors=10,
                         show_edges=True,
                         edge_color="black",
                         edge_line_width=0.1,
                         background="white",
                         window_size=(1100, 800),
                         camera_position=None,
                         save_path=None,
                         plotter=None, return_plotter=False):
    """PyVista off-screen renderer of a scalar field on the surface mesh.

    The colorbar is discretised into ``n_colors`` bins (10 by default).
    ``show_edges=True`` draws the triangle wireframe on top.

    If ``norm`` is provided (e.g. ``matplotlib.colors.TwoSlopeNorm``) the
    per-cell colors are baked through it and a hidden swatch mesh registers a
    faithful scalar bar.

    Always renders off-screen to ``save_path``; there is no interactive
    viewer here.
    """
    import pyvista as pv
    with pv.vtk_verbosity('off'):

        lower_field = np.asarray(lower_field, dtype=float)
        if lower_field.shape[0] != lower_mesh["triangles"].shape[0]:
            raise ValueError("lower_field must match number of lower triangles")
        if upper_mesh is not None and upper_field is not None:
            upper_field = np.asarray(upper_field, dtype=float)
            if upper_field.shape[0] != upper_mesh["triangles"].shape[0]:
                raise ValueError("upper_field must match number of upper triangles")

        if vmin is None:
            vmin = float(lower_field.min())
            if upper_field is not None:
                vmin = min(vmin, float(upper_field.min()))
        if vmax is None:
            vmax = float(lower_field.max())
            if upper_field is not None:
                vmax = max(vmax, float(upper_field.max()))

        own_plotter = plotter is None
        if own_plotter:
            plotter = pv.Plotter(off_screen=True, window_size=window_size)
            plotter.set_background(background)
            try:
                plotter.enable_depth_peeling(number_of_peels=8,
                                            occlusion_ratio=0.0)
            except Exception:
                pass

        sb_args = dict(title=colorbar_label, n_labels=n_colors + 1, fmt="%.3g",
                    vertical=True, position_x=0.88, position_y=0.15,
                    width=0.06, height=0.7)

        use_rgb = norm is not None
        cmap_obj = cm.get_cmap(cmap, n_colors)

        def _add_surface(mesh_dict, field_arr, opacity):
            pd = _mesh_to_polydata(mesh_dict)
            if use_rgb:
                pd.cell_data["RGB"] = _quantize_rgb(field_arr, norm,
                                                    cmap_obj, n_colors)
                plotter.add_mesh(pd, scalars="RGB", rgb=True,
                                opacity=opacity, show_edges=show_edges,
                                edge_color=edge_color,
                                line_width=edge_line_width,
                                edge_opacity=0.3,
                                smooth_shading=False, show_scalar_bar=False)
            else:
                pd.cell_data["field"] = field_arr
                plotter.add_mesh(pd, scalars="field", cmap=cmap_obj,
                                clim=(vmin, vmax), n_colors=n_colors,
                                opacity=opacity, show_edges=show_edges,
                                edge_color=edge_color,
                                line_width=edge_line_width,
                                edge_opacity=0.3,
                                smooth_shading=False, show_scalar_bar=False)

        _add_surface(lower_mesh, lower_field, lower_alpha)
        if upper_mesh is not None and upper_field is not None:
            _add_surface(upper_mesh, upper_field, upper_alpha)

        # Discretised scalar bar via a hidden swatch mesh
        n_swatch = max(2, n_colors + 1)
        swatch_pts = np.column_stack([np.linspace(0, 1, n_swatch),
                                    np.zeros(n_swatch),
                                    np.zeros(n_swatch)])
        swatch_vals = np.linspace(vmin, vmax, n_swatch)
        sw = pv.PolyData(swatch_pts)
        sw.point_data["field"] = swatch_vals
        plotter.add_mesh(sw, scalars="field", cmap=cmap_obj,
                        clim=(vmin, vmax), n_colors=n_colors,
                        opacity=0.0,
                        show_scalar_bar=True, scalar_bar_args=sb_args)

        if camera_position is not None:
            plotter.camera_position = camera_position
        else:
            plotter.view_isometric()
            plotter.camera.zoom(1.3)
        if return_plotter:
            return plotter
        if save_path:
            plotter.screenshot(save_path)
        if own_plotter:
            plotter.close()


# ---------------------------------------------------------------------------
# Flowfield slices
# ---------------------------------------------------------------------------
def _vsq_to_field(vsq, field, gamma, M1):
    """Convert V'^2 to a chosen scalar field. T0 references upstream stagnation."""
    vsq = np.clip(vsq, 0.0, 0.999999)
    Mloc = np.sqrt(2.0 / (gamma - 1.0) * vsq / (1.0 - vsq))
    if field == "mach":
        return Mloc, "Mach number"
    if field == "temperature":
        # T/T_inf = (1 - V'^2) * (1 + (gamma-1)/2 * M1^2)
        T_ratio = (1.0 - vsq) * (1.0 + 0.5 * (gamma - 1.0) * M1**2)
        return T_ratio, r"T/T_inf"
    if field == "density":
        # Through isentropic conical flow downstream of the shock, using
        # T/T0 = 1-V'^2 and p02 (constant in conical region) so rho/rho02 = (1-V'^2)^(1/(g-1)).
        # For the freestream branch (used outside captured region), the same
        # conversion gives rho/rho_inf = (1 - V'^2)/(1 - V'_inf^2) up to constants;
        # we just plot rho/rho_inf-equivalent via Mach-derived (1-V'^2)^(1/(g-1)).
        rho_ratio = (1.0 - vsq) ** (1.0 / (gamma - 1.0))
        # normalise so freestream (M1) maps to 1
        Vsq_inf = 1.0 / (2.0 / ((gamma - 1.0) * M1**2) + 1.0)
        rho_inf = (1.0 - Vsq_inf) ** (1.0 / (gamma - 1.0))
        return rho_ratio / rho_inf, r"rho/rho_inf"
    raise ValueError(f"unknown field '{field}'")

def plot_flowfield_slices_pv(geom, lower_mesh, upper_mesh,
                             x_planes=None, n_grid=180,
                             field="mach",
                             cmap="Spectral",
                             n_colors=10,
                             slice_alpha=0.55,
                             body_lower_alpha=1.0,
                             body_upper_alpha=1.0,
                             shock_line_width=4.0,
                             show_edges=False,
                             background="white",
                             window_size=(1300, 900),
                             camera_position=None,
                             save_path=None):
    """PyVista off-screen renderer of body + flowfield slices.

    Body is opaque; the slice planes are translucent (``slice_alpha``)
    so the body shows through them. Default camera matches the matplotlib
    flowfield view (elev=-12°, azim=-65°).
    """
    import pyvista as pv
    from scipy.interpolate import interp1d
    import matplotlib.tri as mtri
    from aerodynamics import _tm_vsq_profile

    params = geom["parameters"]
    sc = geom["shock_conditions"]
    M1       = params["M1"]
    gamma    = params["gamma"]
    L        = params["L"]
    Rs       = params["Rs"]
    beta_rad = sc["beta_rad"]
    Vr_i     = sc["Vr_i"]
    V_theta_i = sc["V_theta_i"]

    # T-M V'^2(θ) profile and freestream value
    theta_tm, Vsq_tm = _tm_vsq_profile(gamma, beta_rad, Vr_i, V_theta_i)
    Vsq_interp = interp1d(theta_tm[::-1], Vsq_tm[::-1], kind="linear",
                          bounds_error=False,
                          fill_value=(Vsq_tm[-1], Vsq_tm[0]))
    Vsq_inf = 1.0 / (2.0 / ((gamma - 1.0) * M1**2) + 1.0)

    with pv.vtk_verbosity('off'):

        # Body field
        cents = lower_mesh["centroids"]
        theta_lo = np.arctan2(np.sqrt(cents[:, 1]**2 + cents[:, 2]**2), cents[:, 0])
        Vsq_lo = Vsq_interp(theta_lo)
        f_lower, cbar_label = _vsq_to_field(Vsq_lo, field, gamma, M1)
        f_upper_val, _      = _vsq_to_field(np.full(1, Vsq_inf), field, gamma, M1)
        f_upper = np.full(upper_mesh["triangles"].shape[0], float(f_upper_val[0]))

        f_shock, _ = _vsq_to_field(Vsq_tm[:1], field, gamma, M1)
        f_cone,  _ = _vsq_to_field(Vsq_tm[-1:], field, gamma, M1)
        f_inf,   _ = _vsq_to_field(np.full(1, Vsq_inf), field, gamma, M1)
        f_min = min(float(f_shock[0]), float(f_cone[0]), float(f_inf[0]))
        f_max = max(float(f_shock[0]), float(f_cone[0]), float(f_inf[0]))
        f_post = float(f_shock[0])
        if f_min < f_post < f_max:
            norm = colors.TwoSlopeNorm(vmin=f_min, vcenter=f_post, vmax=f_max)
        else:
            norm = colors.Normalize(vmin=f_min, vmax=f_max)

        # Default camera
        if camera_position is None:
            focal = (L / 2.0 + 0.25 * L, 0.0, -Rs / 4.0)
            camera_position = _camera_from_elev_azim(
                elev_deg=-12.0, azim_deg=-65.0,
                focal=focal, distance=1.6 * L,
                view_up=(0.0, 0.0, 1.0),
            )

        # Build the body plotter via plot_scalar_field_pv (we pass through the
        # discrete-cmap+norm so it picks the same colors as the slices).
        plotter = plot_scalar_field_pv(
            lower_mesh, f_lower,
            upper_mesh=upper_mesh, upper_field=f_upper,
            cmap=cmap, colorbar_label=cbar_label,
            vmin=f_min, vmax=f_max, norm=norm,
            n_colors=n_colors,
            lower_alpha=body_lower_alpha, upper_alpha=body_upper_alpha,
            show_edges=show_edges,
            background=background, window_size=window_size,
            camera_position=camera_position,
            return_plotter=True,
        )

        cmap_obj = cm.get_cmap(cmap, n_colors)

        le = geom["leading_edge"]
        X_LE = np.asarray(le["x"], dtype=float)
        Y_LE = np.asarray(le["y"], dtype=float)
        Z_LE = np.asarray(le["z"], dtype=float)
        phi_LE = np.arctan2(Y_LE, -Z_LE)

        if x_planes is None:
            x_planes = [0.45 * L, 0.7 * L, 1.0 * L]

        extent = 1.15 * Rs

        for x_p in x_planes:
            cap = X_LE <= x_p + 1e-12
            if not cap.any():
                continue
            phi_max = float(np.max(np.abs(phi_LE[cap])))
            Rs_xp   = x_p * np.tan(beta_rad)

            # r_body(φ) from lower-surface streamlines
            phi_body, rho_body = [], []
            for ls in geom["lower_surface"]:
                crv = np.asarray(ls["curve"], dtype=float)
                xs = crv[:, 0]
                if xs.min() - 1e-9 <= x_p <= xs.max() + 1e-9:
                    idx = np.argsort(xs)
                    ys = float(np.interp(x_p, xs[idx], crv[idx, 1]))
                    zs = float(np.interp(x_p, xs[idx], crv[idx, 2]))
                    phi_body.append(float(np.arctan2(ys, -zs)))
                    rho_body.append(float(np.hypot(ys, zs)))
            if len(phi_body) < 2:
                continue
            phi_body = np.asarray(phi_body); rho_body = np.asarray(rho_body)
            phi_full = np.concatenate([-phi_body[::-1], phi_body])
            rho_full = np.concatenate([ rho_body[::-1], rho_body])
            order = np.argsort(phi_full)
            phi_full = phi_full[order]; rho_full = rho_full[order]
            _, uniq = np.unique(phi_full, return_index=True)
            rbody_interp = interp1d(phi_full[uniq], rho_full[uniq], kind="linear",
                                    bounds_error=False, fill_value=Rs_xp)

            # r_upper(φ) from LE break points whose X_LE ≤ x_p
            cap_le = X_LE <= x_p + 1e-12
            phi_up = phi_LE[cap_le]
            rho_up = np.hypot(Y_LE[cap_le], Z_LE[cap_le])
            ord_u = np.argsort(phi_up)
            phi_up = phi_up[ord_u]; rho_up = rho_up[ord_u]
            _, uu = np.unique(phi_up, return_index=True)
            rupper_interp = interp1d(phi_up[uu], rho_up[uu], kind="linear",
                                    bounds_error=False, fill_value=Rs_xp)

            # 2-D (y, z) grid on the slice
            y_lin = np.linspace(-extent, extent, n_grid)
            z_lin = np.linspace(-extent, extent / 6, n_grid)
            Y, Z  = np.meshgrid(y_lin, z_lin)
            rho_g   = np.hypot(Y, Z)
            theta_g = np.arctan2(rho_g, x_p)

            Vsq_grid  = Vsq_interp(theta_g)
            F_grid, _ = _vsq_to_field(Vsq_grid, field, gamma, M1)

            yy = Y.ravel(); zz = Z.ravel(); ff = F_grid.ravel()
            triang = mtri.Triangulation(yy, zz)
            tris = triang.triangles
            yc = yy[tris].mean(axis=1)
            zc = zz[tris].mean(axis=1)
            rc = np.hypot(yc, zc)
            phi_c = np.arctan2(yc, -zc)
            rb_c  = rbody_interp(phi_c)
            ru_c  = rupper_interp(phi_c)

            in_capt = (np.abs(phi_c) <= phi_max) & (rc >= rb_c) & (rc <= Rs_xp)
            in_body = (np.abs(phi_c) <= phi_max) & (rc >= ru_c) & (rc < rb_c)
            keep    = ~in_body
            if not keep.any():
                continue

            tris_k = tris[keep]
            f_per  = np.where(in_capt[keep],
                            ff[tris_k].mean(axis=1),
                            float(f_inf[0]))
            rgb = _quantize_rgb(f_per, norm, cmap_obj, n_colors)

            # PolyData of all kept triangles, on the x=x_p plane
            n_tri = tris_k.shape[0]
            verts = np.empty((3 * n_tri, 3), dtype=float)
            verts[:, 0] = x_p
            verts[:, 1] = yy[tris_k].ravel()
            verts[:, 2] = zz[tris_k].ravel()
            idx = np.arange(3 * n_tri, dtype=np.int64).reshape(n_tri, 3)
            faces = np.concatenate(
                [np.full((n_tri, 1), 3, dtype=np.int64), idx], axis=1
            ).ravel()
            slice_pd = pv.PolyData(verts, faces)
            slice_pd.cell_data["RGB"] = rgb
            plotter.add_mesh(slice_pd, scalars="RGB", rgb=True,
                            opacity=slice_alpha, show_scalar_bar=False,
                            lighting=False)

            # Captured shock arc as a thick black 3-D line
            n_arc = 240
            phi_arc = np.linspace(-phi_max, phi_max, n_arc)
            arc_pts = np.column_stack([
                np.full(n_arc, x_p),
                Rs_xp * np.sin(phi_arc),
                -Rs_xp * np.cos(phi_arc),
            ])
            arc_lines = np.concatenate(
                [[n_arc], np.arange(n_arc, dtype=np.int64)]
            )
            arc = pv.PolyData()
            arc.points = arc_pts
            arc.lines = arc_lines
            plotter.add_mesh(arc, color="black", line_width=0.5,
                            render_lines_as_tubes=False, show_scalar_bar=False)

    if save_path:
        plotter.screenshot(save_path)
    plotter.close()


# ---------------------------------------------------------------------------
# Geometry view grid
# ---------------------------------------------------------------------------

def plot_geometry_views_pv(lower_mesh, upper_mesh,
                           style="wireframe",
                           wire_color="black",
                           body_color="#bfbfbf",
                           background="white",
                           window_size=(1200, 900),
                           edge_line_width=1.0,
                           zoom=1.3,
                           save_path=None):
    """Render a 2x2 grid of geometry views (side, front, iso, top)."""
    import pyvista as pv

    if style not in {"wireframe", "shaded"}:
        raise ValueError("style must be 'wireframe' or 'shaded'")

    def _add_body(plotter):
        pd_lo = _mesh_to_polydata(lower_mesh)
        pd_up = _mesh_to_polydata(upper_mesh)
        if style == "wireframe":
            plotter.add_mesh(pd_lo, color=wire_color, style="wireframe",
                             line_width=edge_line_width)
            plotter.add_mesh(pd_up, color=wire_color, style="wireframe",
                             line_width=edge_line_width)
        else:
            plotter.add_mesh(pd_lo, color=body_color, smooth_shading=True,
                             show_edges=False)
            plotter.add_mesh(pd_up, color=body_color, smooth_shading=True,
                             show_edges=False)

    with pv.vtk_verbosity('off'):
        plotter = pv.Plotter(off_screen=True, shape=(2, 2),
                             window_size=window_size)
        plotter.set_background(background)
        plotter.enable_hidden_line_removal(all_renderers=True)

        # Side view (x-z plane)
        plotter.subplot(0, 0)
        _add_body(plotter)
        plotter.view_xz(negative=True)
        plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.add_title('Side', font='arial', color='k', font_size=14)

        # Front view (y-z plane)
        plotter.subplot(0, 1)
        _add_body(plotter)
        plotter.view_yz(negative=True)
        plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.add_title('Front', font='arial', color='k', font_size=14)

        # Isometric view
        plotter.subplot(1, 1)
        _add_body(plotter)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.add_title('Isometric', font='arial', color='k', font_size=14)

        # Top view (x-y plane)
        plotter.subplot(1, 0)
        _add_body(plotter)
        plotter.view_xy(negative=True)
        plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.add_title('Top', font='arial', color='k', font_size=14)

        if save_path: plotter.screenshot(save_path)
        plotter.close()
