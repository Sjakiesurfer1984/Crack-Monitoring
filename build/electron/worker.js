var le=Object.create;var $=Object.defineProperty;var ce=Object.getOwnPropertyDescriptor;var de=Object.getOwnPropertyNames;var me=Object.getPrototypeOf,ue=Object.prototype.hasOwnProperty;var ge=(r,e)=>()=>(e||r((e={exports:{}}).exports,e),e.exports);var fe=(r,e,t,o)=>{if(e&&typeof e=="object"||typeof e=="function")for(let n of de(e))!ue.call(r,n)&&n!==t&&$(r,n,{get:()=>e[n],enumerable:!(o=ce(e,n))||o.enumerable});return r};var I=(r,e,t)=>(t=r!=null?le(me(r)):{},fe(e||!r||!r.__esModule?$(t,"default",{value:r,enumerable:!0}):t,r));var B=ge((Re,z)=>{"use strict";function w(r){if(typeof r!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(r))}function q(r,e){for(var t="",o=0,n=-1,i=0,s,a=0;a<=r.length;++a){if(a<r.length)s=r.charCodeAt(a);else{if(s===47)break;s=47}if(s===47){if(!(n===a-1||i===1))if(n!==a-1&&i===2){if(t.length<2||o!==2||t.charCodeAt(t.length-1)!==46||t.charCodeAt(t.length-2)!==46){if(t.length>2){var c=t.lastIndexOf("/");if(c!==t.length-1){c===-1?(t="",o=0):(t=t.slice(0,c),o=t.length-1-t.lastIndexOf("/")),n=a,i=0;continue}}else if(t.length===2||t.length===1){t="",o=0,n=a,i=0;continue}}e&&(t.length>0?t+="/..":t="..",o=2)}else t.length>0?t+="/"+r.slice(n+1,a):t=r.slice(n+1,a),o=a-n-1;n=a,i=0}else s===46&&i!==-1?++i:i=-1}return t}function pe(r,e){var t=e.dir||e.root,o=e.base||(e.name||"")+(e.ext||"");return t?t===e.root?t+o:t+r+o:o}var L={resolve:function(){for(var e="",t=!1,o,n=arguments.length-1;n>=-1&&!t;n--){var i;n>=0?i=arguments[n]:(o===void 0&&(o=process.cwd()),i=o),w(i),i.length!==0&&(e=i+"/"+e,t=i.charCodeAt(0)===47)}return e=q(e,!t),t?e.length>0?"/"+e:"/":e.length>0?e:"."},normalize:function(e){if(w(e),e.length===0)return".";var t=e.charCodeAt(0)===47,o=e.charCodeAt(e.length-1)===47;return e=q(e,!t),e.length===0&&!t&&(e="."),e.length>0&&o&&(e+="/"),t?"/"+e:e},isAbsolute:function(e){return w(e),e.length>0&&e.charCodeAt(0)===47},join:function(){if(arguments.length===0)return".";for(var e,t=0;t<arguments.length;++t){var o=arguments[t];w(o),o.length>0&&(e===void 0?e=o:e+="/"+o)}return e===void 0?".":L.normalize(e)},relative:function(e,t){if(w(e),w(t),e===t||(e=L.resolve(e),t=L.resolve(t),e===t))return"";for(var o=1;o<e.length&&e.charCodeAt(o)===47;++o);for(var n=e.length,i=n-o,s=1;s<t.length&&t.charCodeAt(s)===47;++s);for(var a=t.length,c=a-s,g=i<c?i:c,u=-1,d=0;d<=g;++d){if(d===g){if(c>g){if(t.charCodeAt(s+d)===47)return t.slice(s+d+1);if(d===0)return t.slice(s+d)}else i>g&&(e.charCodeAt(o+d)===47?u=d:d===0&&(u=0));break}var b=e.charCodeAt(o+d),F=t.charCodeAt(s+d);if(b!==F)break;b===47&&(u=d)}var v="";for(d=o+u+1;d<=n;++d)(d===n||e.charCodeAt(d)===47)&&(v.length===0?v+="..":v+="/..");return v.length>0?v+t.slice(s+u):(s+=u,t.charCodeAt(s)===47&&++s,t.slice(s))},_makeLong:function(e){return e},dirname:function(e){if(w(e),e.length===0)return".";for(var t=e.charCodeAt(0),o=t===47,n=-1,i=!0,s=e.length-1;s>=1;--s)if(t=e.charCodeAt(s),t===47){if(!i){n=s;break}}else i=!1;return n===-1?o?"/":".":o&&n===1?"//":e.slice(0,n)},basename:function(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');w(e);var o=0,n=-1,i=!0,s;if(t!==void 0&&t.length>0&&t.length<=e.length){if(t.length===e.length&&t===e)return"";var a=t.length-1,c=-1;for(s=e.length-1;s>=0;--s){var g=e.charCodeAt(s);if(g===47){if(!i){o=s+1;break}}else c===-1&&(i=!1,c=s+1),a>=0&&(g===t.charCodeAt(a)?--a===-1&&(n=s):(a=-1,n=c))}return o===n?n=c:n===-1&&(n=e.length),e.slice(o,n)}else{for(s=e.length-1;s>=0;--s)if(e.charCodeAt(s)===47){if(!i){o=s+1;break}}else n===-1&&(i=!1,n=s+1);return n===-1?"":e.slice(o,n)}},extname:function(e){w(e);for(var t=-1,o=0,n=-1,i=!0,s=0,a=e.length-1;a>=0;--a){var c=e.charCodeAt(a);if(c===47){if(!i){o=a+1;break}continue}n===-1&&(i=!1,n=a+1),c===46?t===-1?t=a:s!==1&&(s=1):t!==-1&&(s=-1)}return t===-1||n===-1||s===0||s===1&&t===n-1&&t===o+1?"":e.slice(t,n)},format:function(e){if(e===null||typeof e!="object")throw new TypeError('The "pathObject" argument must be of type Object. Received type '+typeof e);return pe("/",e)},parse:function(e){w(e);var t={root:"",dir:"",base:"",ext:"",name:""};if(e.length===0)return t;var o=e.charCodeAt(0),n=o===47,i;n?(t.root="/",i=1):i=0;for(var s=-1,a=0,c=-1,g=!0,u=e.length-1,d=0;u>=i;--u){if(o=e.charCodeAt(u),o===47){if(!g){a=u+1;break}continue}c===-1&&(g=!1,c=u+1),o===46?s===-1?s=u:d!==1&&(d=1):s!==-1&&(d=-1)}return s===-1||c===-1||d===0||d===1&&s===c-1&&s===a+1?c!==-1&&(a===0&&n?t.base=t.name=e.slice(1,c):t.base=t.name=e.slice(a,c)):(a===0&&n?(t.name=e.slice(1,s),t.base=e.slice(1,c)):(t.name=e.slice(a,s),t.base=e.slice(a,c)),t.ext=e.slice(s,c)),a>0?t.dir=e.slice(0,a-1):n&&(t.dir="/"),t},sep:"/",delimiter:":",win32:null,posix:null};L.posix=L;z.exports=L});var j=require("node:worker_threads");var R=I(B()),H="/home/pyodide",x=r=>`${H}/${r}`,C=(r,e)=>r==null?R.default.resolve(H,e):R.default.resolve(x(r),e);function K(r,e){let t=R.default.normalize(e),n=R.default.dirname(t).split("/"),i=[];for(let s of n){i.push(s);let a=i.join("/");if(r.FS.analyzePath(a).exists){if(r.FS.isDir(a))throw new Error(`"${a}" already exists and is not a directory.`);continue}try{r.FS.mkdir(a)}catch(c){throw console.error(`Failed to create a directory "${a}"`),c}}}function O(r,e,t,o){K(r,e),r.FS.writeFile(e,t,o)}function J(r,e,t){K(r,t),r.FS.rename(e,t)}var _e="[",ye="(<=>!~",he=";",be="@",ve=new RegExp(`[${_e+ye+he+be}]`);function Pe(r){return r.split(ve)[0].trim()}function T(r){return r.forEach(t=>{let o;try{o=new URL(t)}catch{return}if(o.protocol==="emfs:"||o.protocol==="file:")throw new Error(`"emfs:" and "file:" protocols are not allowed for the requirement (${t})`)}),r.filter(t=>Pe(t)==="streamlit"?(console.warn(`Streamlit is specified in the requirements ("${t}"), but it will be ignored. A built-in version of Streamlit will be used.`),!1):!0)}async function we(r){let e=typeof process<"u"&&process.versions?.node,t;e?t=(await import("node:path")).sep:t="/";let o=r.slice(0,r.lastIndexOf(t)+1);if(r.endsWith(".mjs")){if(e){let n=await import("node:path"),i=await import("node:url");!r.includes("://")&&n.isAbsolute(r)&&(r=i.pathToFileURL(r).href)}return{scriptURL:r,pyodideIndexURL:o,isESModule:!0}}else return{scriptURL:r,pyodideIndexURL:o,isESModule:!1}}async function V(r,e){let{scriptURL:t,pyodideIndexURL:o,isESModule:n}=await we(r),i;return n?i=(await import(t)).loadPyodide:(importScripts(t),i=self.loadPyodide),i({...e,indexURL:o})}function G(r){r.runPython(`
import micropip
micropip.add_mock_package(
    "pyarrow", "0.0.1",
    modules={
        "pyarrow": """
__version__ = '0.0.1'  # TODO: Update when releasing


class Table:
    @classmethod
    def from_pandas(*args, **kwargs):
        raise NotImplementedError("stlite is not supporting this method.")


class Array:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("stlite is not supporting PyArrow.Array")


class ChunkedArray:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("stlite is not supporting PyArrow.ChunkedArray")
"""
    }
)
`)}function ke(r,e,t){let o=r.pyimport("pyodide"),n=u=>o.code.find_imports(u).toJs(),i=t.map(u=>n(u)),c=Array.from(new Set(i.flat())).filter(u=>!r.runPython(`__import__('importlib').util.find_spec('${u}')`)).map(u=>r._api._import_name_to_package_name.get(u)).filter(u=>u);if(c.length===0)return Promise.resolve();let g=r.loadPackage(c);return e(c,g),g.then()}function N(r,e,t){let o=ke(r,e,t);r.runPython(`
def __set_module_auto_load_promise__(promise):
    from streamlit.runtime.scriptrunner import script_runner
    script_runner.moduleAutoLoadPromise = promise

__set_module_auto_load_promise__`)(o)}var X=async r=>{await r.runPythonAsync(`import jedi
import re
import json
from typing import Dict
from lsprotocol.types import (CompletionItem, CompletionList, CompletionItemKind, Position, Range, TextEdit)
from lsprotocol import converters as lsp_converters
from jedi.api.classes import Completion

def as_completion_item_kind(kind: str):
  match kind:
    case 'class':
      return CompletionItemKind.Class
    case 'function':
      return CompletionItemKind.Function
    case 'instance':
      return CompletionItemKind.Reference
    case 'keyword':
      return CompletionItemKind.Keyword
    case 'module':
      return CompletionItemKind.Module
    case 'param':
      return CompletionItemKind.Variable
    case 'path':
      return CompletionItemKind.File
    case 'property':
      return CompletionItemKind.Property
    case 'statement':
      return CompletionItemKind.Variable
    case _:
      return CompletionItemKind.Text

def as_completion_item_sort_text(item: Completion) -> str:
  """Generate sorting text to arrange items alphabetically,
  ensuring parameters are prioritized first
  and private magic properties come last.
  """
  completion_item_name = item.name
  if completion_item_name is None or completion_item_name.startswith('_'):
     return f"zz{completion_item_name}"
  elif item.type == "param" and completion_item_name.endswith("="):
     return f"aa{completion_item_name}"
  else:
      return f"bb{completion_item_name}"

def as_completion_item(completion: Completion, cursor_range: Range) -> Dict:
  label = completion.name
  return CompletionItem(
      label=label,
      filter_text=label,
      sort_text=as_completion_item_sort_text(completion),
      kind=as_completion_item_kind(completion.type),
      documentation=completion.docstring(raw=True),
      text_edit=TextEdit(range=cursor_range, new_text=label),
  )

def get_text_edit_cursor_range(cursor_code_line: str, current_line_number: int, cursor_offset: int):
  # Match the substring starting from cursor_offset ex: math<cursor>co, match co
  matched_words = re.search(r'\\b\\w+\\b', cursor_code_line[cursor_offset :])

  # Determine the length of the matched word characters
  word_after_cursor_length = len(matched_words.group()) if matched_words else 0

  # This will tell to code editors which text to edit/replace
  return Range(
    start=Position(
        line=current_line_number, character=cursor_offset
    ),
    end=Position(
        line=current_line_number,
        character=cursor_offset + word_after_cursor_length,
    ),
  )

def get_code_completions(code: str, current_line_number: int, cursor_offset: int):

  jedi_language_server = jedi.Script(code)

  # jedi returns a zero-based array with lines
  jedi_line_number_index = current_line_number -1

  # In case if we are not getting any results back or the offset is wrong
  # Just return empty list
  if jedi_line_number_index >= len(jedi_language_server._code_lines):
   return json.dumps({ "items": []})

  jedi_completions_list = jedi_language_server.complete(
      current_line_number,
      cursor_offset,
      fuzzy=False,
  )

  code_at_cursor = jedi_language_server._code_lines[jedi_line_number_index]
  cursor_range = get_text_edit_cursor_range(code_at_cursor, current_line_number, cursor_offset)

  # Convert jedi completion items as completion items compatible in language server
  suggestions = CompletionList(
    is_incomplete=False,
    items=list(as_completion_item(completion, cursor_range) for completion in jedi_completions_list))

  # Convert results to JSON so that we can use it in the worker
  converter = lsp_converters.get_converter()
  return json.dumps(converter.unstructure(suggestions, unstructure_as=CompletionList))
`)},Q=async(r,e)=>{let t;try{if(t=e.globals.get("get_code_completions"),!t)return console.error("Can not generate suggestions list, the get_code_completions function is not defined"),{items:[]};let o=t(r.code,r.line,r.column);return o?JSON.parse(o):{items:[]}}catch(o){return console.error(o),{items:[]}}finally{t&&t.constructor.name==="PyProxy"&&t.destroy()}};var Y=async(r,e)=>{try{console.debug("Importing jedi Interpreter"),await e.install.callKwargs(["jedi","lsprotocol"],{keep_going:!0}),await X(r)}catch(t){console.error("Error while importing jedi",t)}};var D=null;async function Se(r,e,t,o,n){let{entrypoint:i,files:s,archives:a,requirements:c,prebuiltPackageNames:g,wheels:u,pyodideUrl:d=r,streamlitConfig:b,idbfsMountpoints:F,nodefsMountpoints:v,moduleAutoLoad:E,env:h,languageServer:m}=t,p=T(c);D?(n("Pyodide is already loaded."),console.debug("Pyodide is already loaded.")):(n("Loading Pyodide."),console.debug("Loading Pyodide."),D=V(d,{stdout:console.log,stderr:console.error}),u&&(p.unshift(u.streamlit),p.unshift(u.stliteLib)),console.debug("Loaded Pyodide"));let l=await D;if(h){console.debug("Setting environment variables",h);let f=l.pyimport("os");f.environ.update(l.toPy(h)),console.debug("Set environment variables",f.environ)}let y=!1;F&&(y=!0,F.forEach(f=>{l.FS.mkdir(f),l.FS.mount(l.FS.filesystems.IDBFS,{},f)}),await new Promise((f,_)=>{l.FS.syncfs(!0,P=>{P?_(P):f()})})),v&&Object.entries(v).forEach(([f,_])=>{l.FS.mkdir(f),l.FS.mount(l.FS.filesystems.NODEFS,{root:_},f)}),n("Mounting files.");let k=[];await Promise.all(Object.keys(s).map(async f=>{let _=s[f];f=C(e,f);let P;"url"in _?(console.debug(`Fetch a file from ${_.url}`),P=await fetch(_.url).then(A=>A.arrayBuffer()).then(A=>new Uint8Array(A))):P=_.data,console.debug(`Write a file "${f}"`),O(l,f,P,s.opts),f.endsWith(".py")&&k.push(f)})),n("Unpacking archives."),await Promise.all(a.map(async f=>{let _;"url"in f?(console.debug(`Fetch an archive from ${f.url}`),_=await fetch(f.url).then(ae=>ae.arrayBuffer())):_=f.buffer;let{format:P,options:A}=f;console.debug("Unpack an archive",{format:P,options:A}),l.unpackArchive(_,P,A)})),await l.loadPackage("micropip");let S=l.pyimport("micropip");if(n("Mocking some packages."),console.debug("Mock pyarrow"),G(l),console.debug("Mocked pyarrow"),n("Installing packages."),console.debug("Installing the prebuilt packages:",g),await l.loadPackage(g),console.debug("Installed the prebuilt packages"),console.debug("Installing the requirements:",p),await S.install.callKwargs(p,{keep_going:!0}),console.debug("Installed the requirements"),E){let f=k.map(_=>l.FS.readFile(_,{encoding:"utf8"}));N(l,o,f)}await l.runPythonAsync(`
import importlib
importlib.invalidate_caches()
`),n("Loading streamlit package."),console.debug("Loading the Streamlit package"),await l.runPythonAsync(`
import streamlit.runtime
  `),console.debug("Loaded the Streamlit package"),n("Setting up the loggers."),console.debug("Setting the loggers"),await l.runPythonAsync(`
import logging
import streamlit.logger

streamlit.logger.get_logger = logging.getLogger
streamlit.logger.setup_formatter = None
streamlit.logger.update_formatter = lambda *a, **k: None
streamlit.logger.set_log_level = lambda *a, **k: None

for name in streamlit.logger._loggers.keys():
    if name == "root":
        name = "streamlit"
    logger = logging.getLogger(name)
    logger.propagate = True
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)

streamlit.logger._loggers = {}
`);let M=(f,_)=>{f>=40?console.error(_):f>=30?console.warn(_):f>=20?console.info(_):console.debug(_)},te=l.runPython(`
def __setup_loggers__(streamlit_level, streamlit_message_format, callback):
    class JsHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            callback(record.levelno, msg)


    root_message_format = "%(levelname)s:%(name)s:%(message)s"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_formatter = logging.Formatter(root_message_format)
    root_handler = JsHandler()
    root_handler.setFormatter(root_formatter)
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.DEBUG)

    streamlit_logger = logging.getLogger("streamlit")
    streamlit_logger.propagate = False
    streamlit_logger.handlers.clear()
    streamlit_formatter = logging.Formatter(streamlit_message_format)
    streamlit_handler = JsHandler()
    streamlit_handler.setFormatter(streamlit_formatter)
    streamlit_logger.addHandler(streamlit_handler)
    streamlit_logger.setLevel(streamlit_level.upper())

__setup_loggers__`),re=(b?.["logger.level"]??"INFO").toString(),oe=b?.["logger.messageFormat"]??"%(asctime)s %(message)s";if(te(re,oe,M),console.debug("Set the loggers"),n("Mocking some Streamlit functions for the browser environment."),console.debug("Mocking some Streamlit functions"),await l.runPythonAsync(`
import streamlit

def is_cacheable_msg(msg):
  return False

streamlit.runtime.runtime.is_cacheable_msg = is_cacheable_msg
`),console.debug("Mocked some Streamlit functions"),y){n("Setting up the IndexedDB filesystem synchronizer."),console.debug("Setting up the IndexedDB filesystem synchronizer");let f=!1,_=()=>{console.debug("The script has finished. Syncing the filesystem."),f||(f=!0,l.FS.syncfs(!1,A=>{f=!1,A&&console.error(A)}))};(await l.runPython(`
def __setup_script_finished_callback__(callback):
    from streamlit.runtime.app_session import AppSession
    from streamlit.runtime.scriptrunner import ScriptRunnerEvent

    def wrap_app_session_on_scriptrunner_event(original_method):
        def wrapped(self, *args, **kwargs):
            if "event" in kwargs:
                event = kwargs["event"]
                if event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS or event == ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN or event == ScriptRunnerEvent.SHUTDOWN:
                    callback()
            return original_method(self, *args, **kwargs)
        return wrapped

    AppSession._on_scriptrunner_event = wrap_app_session_on_scriptrunner_event(AppSession._on_scriptrunner_event)

__setup_script_finished_callback__`))(_),console.debug("Set up the IndexedDB filesystem synchronizer")}m&&(n("Importing Language Server"),await Y(l,S)),n("Booting up the Streamlit server."),console.debug("Setting up the Streamlit configuration");let ne=await l.runPython(`
def __bootstrap__(main_script_path, flag_options, shared_worker_mode):
    from stlite_lib.bootstrap import load_config_options, prepare

    load_config_options(flag_options, shared_worker_mode)

    prepare(main_script_path, [])

__bootstrap__`),U=C(e,i),se={"browser.gatherUsageStats":!1,...b,"runner.fastReruns":!1},ie=e!=null;ne(U,l.toPy(se),ie),console.debug("Set up the Streamlit configuration"),console.debug("Booting up the Streamlit server");let W=l.pyimport("stlite_lib.server.Server")(U,e?x(e):null);return await W.start(),console.debug("Booted up the Streamlit server"),{pyodide:l,httpServer:W,micropip:S,initData:t}}function Z(r,e,t,o){function n(c){e({type:"event:progress",data:{message:c}})}let i=(c,g)=>{let u=new MessageChannel;e({type:"event:moduleAutoLoad",data:{packagesToLoad:c}},[u.port2]),g.then(d=>{u.port1.postMessage({type:"moduleAutoLoad:success",data:{loadedPackages:d}}),u.port1.close()}).catch(d=>{throw u.port1.postMessage({type:"moduleAutoLoad:error",error:d}),u.port1.close(),d})},s=null,a=async c=>{let g=c.data;if(g.type==="initData"){let m=g.data,p={...t,...m};console.debug("Initial data",p),s=Se(r,o,p,i,n),s.then(()=>{e({type:"event:loaded"})}).catch(l=>{console.error(l),e({type:"event:error",data:{error:l}})});return}if(!s)throw new Error("Pyodide initialization has not been started yet.");let u=await s,d=u.pyodide,b=u.httpServer,F=u.micropip,{moduleAutoLoad:v}=u.initData,E=c.ports[0];function h(m){E.postMessage(m)}try{switch(g.type){case"reboot":{console.debug("Reboot the Streamlit server",g.data);let{entrypoint:m}=g.data;b.stop(),console.debug("Booting up the Streamlit server");let p=C(o,m);b=d.pyimport("stlite_lib.server.Server")(p),b.start(),console.debug("Booted up the Streamlit server"),h({type:"reply"});break}case"websocket:connect":{console.debug("websocket:connect",g.data);let{path:m}=g.data;b.start_websocket(m,(p,l)=>{if(l){let y=p;try{let k=y.toJs(),S=k.buffer.slice(k.byteOffset,k.byteOffset+k.byteLength);e({type:"websocket:message",data:{payload:S}},[S])}finally{y.destroy()}}else e({type:"websocket:message",data:{payload:p}})}),h({type:"reply"});break}case"websocket:send":{console.debug("websocket:send",g.data);let{payload:m}=g.data;b.receive_websocket_from_js(m);break}case"http:request":{console.debug("http:request",g.data);let{request:m}=g.data,p=(l,y,k)=>{let S=new Map(y.toJs()),M=k.toJs();console.debug({statusCode:l,headers:S,body:M}),h({type:"http:response",data:{response:{statusCode:l,headers:S,body:M}}})};b.receive_http_from_js(m.method,decodeURIComponent(m.path),m.headers,m.body,p);break}case"file:write":{let{path:m,data:p,opts:l}=g.data,y=C(o,m);v&&typeof p=="string"&&y.endsWith(".py")&&(console.debug(`Auto install the requirements in ${y}`),N(d,i,[p])),console.debug(`Write a file "${y}"`),O(d,y,p,l),h({type:"reply"});break}case"file:rename":{let{oldPath:m,newPath:p}=g.data,l=C(o,m),y=C(o,p);console.debug(`Rename "${l}" to ${y}`),J(d,l,y),h({type:"reply"});break}case"file:unlink":{let{path:m}=g.data,p=C(o,m);console.debug(`Remove "${p}`),d.FS.unlink(p),h({type:"reply"});break}case"file:read":{let{path:m,opts:p}=g.data;console.debug(`Read "${m}"`);let l=d.FS.readFile(m,p);h({type:"reply:file:read",data:{content:l}});break}case"install":{let{requirements:m}=g.data,p=T(m);console.debug("Install the requirements:",p),await F.install.callKwargs(p,{keep_going:!0}).then(()=>{console.debug("Successfully installed"),h({type:"reply"})});break}case"setEnv":{let{env:m}=g.data;d.pyimport("os").environ.update(d.toPy(m)),console.debug("Successfully set the environment variables",m),h({type:"reply"});break}case"language-server:code_completion":{let m=await Q(g.data,d);h({type:"reply:language-server:code_completion",data:m});break}}}catch(m){if(console.error(m),!(m instanceof Error))throw m;let p=new Error(m.message);p.name=m.name,p.stack=m.stack,h({type:"reply",error:p})}};return e({type:"event:start"}),a}function ee(){let r=process.env.NODEFS_MOUNTPOINTS;if(!r)return;let e;try{e=JSON.parse(r)}catch{console.error(`Failed to parse NODEFS_MOUNTPOINTS as JSON: ${r}`);return}if(typeof e!="object"){console.error(`NODEFS_MOUNTPOINTS is not an object: ${r}`);return}if(Array.isArray(e)){console.error(`NODEFS_MOUNTPOINTS is an array: ${r}`);return}if(Object.keys(e).some(t=>typeof t!="string")){console.error(`NODEFS_MOUNTPOINTS has non-string keys: ${r}`);return}if(Object.values(e).some(t=>typeof t!="string")){console.error(`NODEFS_MOUNTPOINTS has non-string values: ${r}`);return}return e}var Ae=r=>{console.debug("[worker thread] postMessage from worker",r),j.parentPort?.postMessage(r)},Ce=Z(process.env.PYODIDE_URL,Ae,{nodefsMountpoints:ee()});j.parentPort?.on("message",({data:r,port:e})=>{console.debug("[worker thread] parentPort.onMessage",{data:r,port:e}),Ce({data:r,ports:[e]})});
