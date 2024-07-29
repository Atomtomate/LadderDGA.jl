using Term
using Term.Layout
import Term.Grid: grid

wd1 = min(floor(Int,Term.Consoles.console_width()*0.6666), 100)
wd2 = min(floor(Int,Term.Consoles.console_width()*0.3333),  43)
hd0 = min(floor(Int,Term.Consoles.console_height()*1.0), 20)
hd1 = min(floor(Int,Term.Consoles.console_height()*0.9), 19)
hd2 = min(floor(Int,Term.Consoles.console_height()*0.5), 11)
hd3 = min(floor(Int,Term.Consoles.console_height()*0.1), 2)


input_controls = Dict(
    'q' => Term.LiveWidgets.quit,
    Term.Esc() => Term.LiveWidgets.quit,
)

log_panel = Term.Pager("", title="Log", height=hd1, width=wd1,controls=input_controls)
status_panel =  Term.Pager("", title="Status", height=hd2, width=wd2,controls=input_controls)
vars_panel =  Term.Pager("", title="Vars", height=hd2, width=wd2,controls=input_controls)
input_box = Term.InputBox(controls=input_controls)

col1 = :(vstack(
    LW($(hd1), $(wd1)),
    IW($(hd3), $(wd1))
))
col2 = :(vstack(
    B($(hd2), $(wd2)),
    C($(hd2), $(wd2))
    ))
layout = :($col1 * $col2)
widgets = Dict(:LW => log_panel, :IW => input_box, :B => vars_panel, :C => status_panel)

app = Term.App(layout; widgets=widgets, 
               transition_rules = Dict(
               #ArrowRight() => Dict(:a => :b1, :b1=>:c, :b2=>:c),
               #ArrowLeft() => Dict(:c => :b1, :b1=>:a, :b2=>:a),
                Term.ArrowDown() => Dict(:LW => :IW),
                Term.ArrowUp() => Dict(:IW => :LW),
                )) 
Term.Consoles.hide_cursor()
Term.LiveWidgets.set_active(app, :LW)


Term.play(app)
Term.Consoles.show_cursor()