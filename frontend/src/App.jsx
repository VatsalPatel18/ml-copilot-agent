import React, { useState, useCallback, useEffect, createContext, useContext, useRef } from 'react';
import { FileText, Settings, Play, BarChart2, Cpu, FlaskConical, List, FolderOpen, BrainCircuit, FileQuestion, X, Loader2, Bot, KeyRound, Server, RefreshCw, CheckCircle, AlertTriangle, FolderPlus, BookOpen, ChevronDown, ChevronRight, Trash2, BarChartHorizontal, PlusCircle, Edit } from 'lucide-react'; // Added Edit icon

// --- Shadcn/ui Component Mocks (Using basic HTML elements) ---
// ... (Mocks for Button, Input, Label, Textarea, Card, Select, MultiSelect, ScrollArea remain the same) ...
const Button = ({ children, variant = 'default', size = 'default', className = '', disabled, ...props }) => ( <button className={`inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 whitespace-nowrap ${variant === 'destructive' ? 'bg-red-500 text-white hover:bg-red-500/90' : ''} ${variant === 'outline' ? 'border border-input bg-background hover:bg-accent hover:text-accent-foreground' : ''} ${variant === 'secondary' ? 'bg-gray-200 text-secondary-foreground hover:bg-gray-300/80' : ''} ${variant === 'ghost' ? 'hover:bg-accent hover:text-accent-foreground' : ''} ${variant === 'link' ? 'text-blue-600 underline-offset-4 hover:underline' : ''} ${variant === 'default' ? 'bg-blue-600 text-white hover:bg-blue-700/90' : ''} ${size === 'sm' ? 'h-9 px-3' : ''} ${size === 'lg' ? 'h-11 px-8' : ''} ${size === 'icon' ? 'h-10 w-10' : ''} ${size === 'default' ? 'h-10 px-4 py-2' : ''} ${className}`} disabled={disabled} {...props}> {children} </button> );
const Input = ({ className = '', disabled, ...props }) => ( <input className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`} disabled={disabled} {...props} /> );
const Label = ({ className = '', ...props }) => ( <label className={`block text-sm font-medium text-gray-700 leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 ${className}`} {...props} /> );
const Textarea = ({ className = '', disabled, ...props }) => ( <textarea className={`flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`} disabled={disabled} {...props} /> );
const Card = ({ className = '', children }) => <div className={`rounded-lg border bg-white text-card-foreground shadow-sm ${className}`}>{children}</div>;
const CardHeader = ({ className = '', children }) => <div className={`flex flex-col space-y-1.5 p-4 md:p-6 ${className}`}>{children}</div>;
const CardTitle = ({ className = '', children }) => <h3 className={`text-lg font-semibold leading-none tracking-tight ${className}`}>{children}</h3>;
const CardDescription = ({ className = '', children }) => <p className={`text-sm text-gray-500 ${className}`}>{children}</p>;
const CardContent = ({ className = '', children }) => <div className={`p-4 md:p-6 pt-0 ${className}`}>{children}</div>;
const CardFooter = ({ className = '', children }) => <div className={`flex items-center p-4 md:p-6 pt-0 ${className}`}>{children}</div>;
const Select = ({ children, className = '', disabled, ...props }) => ( <select className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`} disabled={disabled} {...props}> {children} </select> );
const SelectTrigger = ({ children }) => children; const SelectValue = ({ placeholder }) => placeholder; const SelectContent = ({ children }) => children; const SelectItem = ({ children, value, ...props }) => <option value={value} {...props}>{children}</option>;
const MultiSelect = ({ options, selected, onChange, placeholder, className, disabled }) => ( <div className={`p-2 border rounded-md bg-gray-50 min-h-[40px] ${className} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}> <span className="text-sm text-gray-600">{selected.length > 0 ? selected.join(', ') : placeholder}</span> <p className="text-xs text-gray-400 mt-1">(Multi-select UI placeholder)</p> </div> );
const ScrollArea = React.forwardRef(({ children, className = '' }, ref) => ( <div ref={ref} className={`overflow-auto ${className}`}>{children}</div> ));
const toast = { info: (message) => console.log("INFO:", message), success: (message) => console.log("SUCCESS:", message), error: (message) => console.error("ERROR:", message), warning: (message) => console.warn("WARNING:", message), }; const Toaster = () => null;

// --- App Context ---
// ... (AppContext, AppProvider, useAppContext remain the same) ...
const AppContext = createContext();
export const AppProvider = ({ children }) => {
    const [currentProject, setCurrentProject] = useState(null); const [projects, setProjects] = useState([]); const [workingDirectory, setWorkingDirectory] = useState('.'); const [llmConfig, setLlmConfig] = useState({ provider: 'openai', apiKey: '', apiModel: 'gpt-4o', ollamaModel: 'llama3', ollamaEndpoint: 'http://localhost:11434', }); const [availableOllamaModels, setAvailableOllamaModels] = useState([]); const [configStatus, setConfigStatus] = useState({ llm: 'pending', cwd: 'pending' });
    const updateWorkingDirectory = useCallback((newDir) => setWorkingDirectory(newDir), []); const updateLlmConfig = useCallback((newConfig) => setLlmConfig(prev => ({ ...prev, ...newConfig })), []); const updateConfigStatus = useCallback((type, status) => setConfigStatus(prev => ({ ...prev, [type]: status })), []);
    const loadProjects = useCallback(async (addLog) => { addLog("Fetching project list..."); try { const data = await callBackendApi('/projects'); setProjects(data.projects || []); addLog(`Found ${data.projects?.length || 0} projects.`); } catch (error) { addLog(`Error fetching projects: ${error.message}`, 'error'); setProjects([]); } }, []);
    const selectProject = useCallback(async (projectName, addLog) => { if (!projectName) { setCurrentProject(null); addLog("No project selected."); return; } addLog(`Loading project: ${projectName}...`); try { const response = await callBackendApi('/load_project', 'POST', { path: projectName }); if (response.success) { setCurrentProject(projectName); addLog(`Project '${projectName}' loaded successfully.`, 'success'); toast.success(`Project '${projectName}' loaded.`); } else { throw new Error(response.message || "Failed to load project"); } } catch (error) { addLog(`Error loading project '${projectName}': ${error.message}`, 'error'); toast.error(`Failed to load project: ${error.message}`); setCurrentProject(null); } }, []);
    const createProject = useCallback(async (projectName, addLog) => { addLog(`Creating project: ${projectName}...`); try { const response = await callBackendApi('/create_project', 'POST', { name: projectName }); if (response.success) { addLog(`Project '${projectName}' created.`, 'success'); toast.success(`Project '${projectName}' created.`); loadProjects(addLog); selectProject(projectName, addLog); } else { throw new Error(response.message || "Failed to create project"); } } catch (error) { addLog(`Error creating project '${projectName}': ${error.message}`, 'error'); toast.error(`Project creation failed: ${error.message} (Backend endpoint may need implementation)`); throw error; /* Re-throw error so UI knows */ } }, [loadProjects, selectProject]); // Added throw
    const value = { currentProject, selectProject, projects, loadProjects, createProject, workingDirectory, updateWorkingDirectory, llmConfig, updateLlmConfig, availableOllamaModels, setAvailableOllamaModels, configStatus, updateConfigStatus, }; return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};
export const useAppContext = () => useContext(AppContext);


// --- API Helper (Using Fetch) ---
// ... (callBackendApi function remains the same - includes mock for /create_project) ...
const API_BASE_URL = 'http://localhost:8000/api';
async function callBackendApi(endpoint, method = 'GET', body = null) { console.log(`Calling backend: ${method} ${API_BASE_URL}${endpoint}`, body); /* MOCK IMPLEMENTATION START */ await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 500)); try { if (endpoint === '/create_project' && method === 'POST') { const projectName = body?.name; if (!projectName) throw new Error("Project name is required for creation."); console.log(`Mock creating project: ${projectName}`); return { success: true, message: `Project '${projectName}' created (mock).` }; } if (endpoint === '/projects' && method === 'GET') { return { projects: ["project_alpha", "example_project", "newly_created_mock"] }; } if (endpoint === '/load_project' && method === 'POST') { const projectName = body?.path; if (!projectName) throw new Error("Project name required to load."); console.log(`Mock loading project: ${projectName}`); return { success: true, message: `Project '${projectName}' loaded (mock).` }; } switch (endpoint) { case '/files': const mockFiles = { '.': ['README.md', 'results/', 'models/', 'data/', 'plots/'], 'data/': ['preprocessed_data.csv', 'raw_data.csv', 'other_data.tsv'], 'results/': ['evaluation.txt', 'metrics.json', 'autopilot_metrics_xyz.json'], 'models/': ['model.pkl', 'model_v2.joblib'], 'plots/': ['plot_abc.png', 'autopilot_boxplot_xyz.png'] }; const pathKey = method === 'GET' ? (new URLSearchParams(body)).get('path') || '.' : body?.path || '.'; return { files: mockFiles[pathKey] || [], current_directory: pathKey }; case '/set_working_directory': if (!body?.path) throw new Error("Path is required."); return { success: true, message: `Working directory set to ${body.path}`, path: body.path }; case '/get_columns': const filePath = method === 'GET' ? (new URLSearchParams(body)).get('file_path') : body?.file_path; if (!filePath || !filePath.includes('.data') && !filePath.includes('.csv')) return { columns: [] }; return { columns: ['feature1', 'feature2', 'categorical_feature', 'target_label', 'id_col', 'another_numeric'] }; case '/preprocess': return { message: `Preprocessing started for ${body.dataset_path}. Output: ${body.save_path}` }; case '/train': return { message: `Training started using ${body.data_path}. Model: ${body.model_save_path}` }; case '/evaluate': return { message: `Evaluation started for ${body.model_path}. Results: ${body.evaluation_save_path}`, results: { accuracy: Math.random() * 0.1 + 0.88, precision: Math.random() * 0.1 + 0.85, recall: Math.random() * 0.1 + 0.89, f1: Math.random() * 0.1 + 0.87, auc: Math.random() * 0.05 + 0.94 } }; case '/plot': const plotId = Date.now(); return { message: `Plot generated: "${body.additional_instructions || 'default'}"`, plot_path: `/mock/path/to/plots/plot_${plotId}.png` }; case '/custom': return { message: `Executing: "${body.instruction}"`, result: `Mock result for instruction.` }; case '/ollama_models': const models = ["llama3:latest", "mistral:latest"]; if (Math.random() < 0.1) throw new Error("Mock: Failed to connect to Ollama"); return { models: models.map(m => ({ name: m })) }; case '/set_llm_config': return { success: true, message: `LLM configuration updated to use ${body.provider}` }; case '/status': return { status: 'idle', message: 'Ready (Mock Backend).', working_directory: '.', config_status: { llm: 'ok', cwd: 'ok' } }; case '/autopilot': const apPlotId = Date.now(); return { message: `Auto-Pilot finished ${body.iterations} iterations.`, plot_path: `/mock/path/to/plots/autopilot_boxplot_${apPlotId}.png` }; default: throw new Error(`Mock API endpoint ${endpoint} not found`); } } catch (error) { console.error("Mock API call failed:", error); toast.error(`API Error: ${error.message}`); throw error; } /* MOCK IMPLEMENTATION END */
    /* // --- ACTUAL FETCH Implementation --- ... */
}


// --- Components ---

// Project Panel (Fixed Create UI - Unchanged from previous fix)
const ProjectPanel = ({ addLog }) => { /* ... same as previous fixed version ... */
    const { currentProject, selectProject, projects, loadProjects, createProject } = useAppContext(); const [newProjectName, setNewProjectName] = useState(''); const [isCreating, setIsCreating] = useState(false); const [createLoading, setCreateLoading] = useState(false); useEffect(() => { loadProjects(addLog); }, [loadProjects, addLog]); const handleCreateToggle = () => { setIsCreating(!isCreating); setNewProjectName(''); }; const handleCreateSubmit = async () => { if (newProjectName.trim()) { setCreateLoading(true); try { await createProject(newProjectName.trim(), addLog); setIsCreating(false); setNewProjectName(''); } catch (error) { /* Error handled in createProject */ } finally { setCreateLoading(false); } } else { toast.warning("Please enter a name for the new project."); } }; return ( <div className="p-2 border-b space-y-2"> <div> <Label htmlFor="projectSelect" className="text-xs font-medium text-gray-500 mb-1">Project</Label> <div className="flex gap-1"> <Select id="projectSelect" value={currentProject || ""} onChange={(e) => selectProject(e.target.value, addLog)} className="flex-grow h-8 text-sm px-2 py-1" disabled={isCreating} > <SelectItem value="" disabled>Select Project</SelectItem> {projects.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)} </Select> <Button onClick={() => loadProjects(addLog)} variant="outline" size="icon" className="h-8 w-8 flex-shrink-0" disabled={isCreating}> <RefreshCw className="h-4 w-4" /> </Button> </div> </div> <div> {isCreating ? ( <div className="space-y-1 p-2 border rounded bg-gray-100"> <Label htmlFor="newProjectName" className="text-xs font-medium text-gray-600">New Project Name</Label> <div className="flex gap-1"> <Input id="newProjectName" type="text" value={newProjectName} onChange={e => setNewProjectName(e.target.value)} placeholder="Enter name..." className="flex-grow h-8 text-sm px-2 py-1" disabled={createLoading} /> <Button onClick={handleCreateSubmit} size="sm" className="h-8 text-xs flex-shrink-0" disabled={createLoading}> {createLoading ? <Loader2 className="h-4 w-4 animate-spin"/> : "Create"} </Button> </div> <Button onClick={handleCreateToggle} variant="link" size="sm" className="text-xs p-0 h-auto" disabled={createLoading}>Cancel</Button> </div> ) : ( <Button onClick={handleCreateToggle} variant="outline" size="sm" className="w-full h-8 text-xs"> <PlusCircle className="h-4 w-4 mr-1"/> Create New Project </Button> )} {!isCreating && <p className="text-xs text-gray-400 mt-1">Note: Project creation requires backend implementation.</p>} </div> </div> );
};


// Sidebar Navigation (Unchanged)
const Sidebar = ({ currentView, setView, addLog }) => { /* ... same as previous ... */
     const { configStatus } = useAppContext(); const navItems = [ { id: 'files', label: 'File Explorer', icon: FolderOpen }, { id: 'preprocess', label: 'Preprocess Data', icon: FlaskConical }, { id: 'train', label: 'Train Model', icon: BrainCircuit }, { id: 'evaluate', label: 'Evaluate Model', icon: BarChart2 }, { id: 'plot', label: 'Plot Results', icon: BarChartHorizontal }, { id: 'custom', label: 'Custom Instruction', icon: Cpu }, { id: 'autopilot', label: 'Auto-Pilot', icon: Play }, ]; const getStatusIcon = (status) => { switch (status) { case 'ok': return <CheckCircle className="h-4 w-4 text-green-500" />; case 'error': return <AlertTriangle className="h-4 w-4 text-red-500" />; default: return <Loader2 className="h-4 w-4 text-yellow-500 animate-spin" />; } }; return ( <aside className="w-64 bg-gray-50 border-r border-gray-200 p-0 flex flex-col"> <div className="flex items-center gap-2 p-4 border-b"> <Bot size={28} className="text-blue-600" /> <h1 className="text-xl font-semibold text-gray-800">ML Copilot</h1> </div> <ProjectPanel addLog={addLog}/> <nav className="flex-grow space-y-1 p-4"> {navItems.map(item => ( <Button key={item.id} variant={currentView === item.id ? 'secondary' : 'ghost'} className={`w-full justify-start ${currentView === item.id ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}`} onClick={() => setView(item.id)}> <item.icon className="mr-2 h-4 w-4" /> {item.label} </Button> ))} </nav> <div className="p-4 border-t mt-auto"> <Button variant={currentView === 'settings' ? 'secondary' : 'ghost'} className={`w-full justify-start ${currentView === 'settings' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}`} onClick={() => setView('settings')}> <Settings className="mr-2 h-4 w-4" /> Settings <div className="ml-auto flex items-center gap-1"> {getStatusIcon(configStatus.cwd)} {getStatusIcon(configStatus.llm)} </div> </Button> </div> </aside> );
};

// Main Content Area Router (Unchanged)
const MainContent = ({ currentView, addLog, clearLogs }) => { /* ... same as previous ... */
     const { currentProject } = useAppContext(); const projectRequiredViews = ['files', 'preprocess', 'train', 'evaluate', 'plot', 'custom', 'autopilot']; if (!currentProject && projectRequiredViews.includes(currentView)) { return ( <Card className="m-4"> <CardHeader><CardTitle>No Project Selected</CardTitle></CardHeader> <CardContent><p>Please select or create a project from the sidebar panel to continue.</p></CardContent> </Card> ); } switch (currentView) { case 'files': return <FilesView addLog={addLog} />; case 'preprocess': return <PreprocessView addLog={addLog} />; case 'train': return <TrainView addLog={addLog} />; case 'evaluate': return <EvaluateView addLog={addLog} />; case 'plot': return <PlotView addLog={addLog} />; case 'custom': return <CustomInstructionView addLog={addLog} />; case 'autopilot': return <AutoPilotView addLog={addLog} />; case 'settings': return <SettingsView addLog={addLog} />; default: return ( <Card className="m-4"> <CardHeader><CardTitle>Welcome to ML Copilot Agent GUI</CardTitle></CardHeader> <CardContent> <p>Select a project, configure settings, or choose an action from the sidebar.</p> <Button onClick={clearLogs} variant="outline" className="mt-4">Clear Logs</Button> </CardContent> </Card> ); }
};

// --- View Components ---

// FilesView (Unchanged)
const FilesView = ({ addLog }) => { /* ... same as previous ... */
    const { workingDirectory } = useAppContext(); const [files, setFiles] = useState([]); const [loading, setLoading] = useState(false); const [error, setError] = useState(null); const [currentPath, setCurrentPath] = useState('.'); const fetchFiles = useCallback(async (relativePath) => { setLoading(true); setError(null); addLog(`Fetching file list for path: ${relativePath} (relative to ${workingDirectory})`); try { const data = await callBackendApi('/files', 'GET', { path: relativePath }); setFiles(data.files || []); setCurrentPath(data.current_directory || relativePath); addLog(`Found ${data.files?.length || 0} items in ${data.current_directory || relativePath}.`); } catch (err) { setError(err.message); addLog(`Error fetching files for ${relativePath}: ${err.message}`, 'error'); toast.error("Failed to load files."); } finally { setLoading(false); } }, [addLog, workingDirectory]); useEffect(() => { setCurrentPath('.'); fetchFiles('.'); }, [fetchFiles, workingDirectory]); const handleItemClick = (item) => { if (item.endsWith('/')) { const dirName = item.slice(0, -1); const newPath = currentPath === '.' ? dirName : `${currentPath}/${dirName}`; fetchFiles(newPath); } else { addLog(`Selected file: ${item}`); } }; const handleGoUp = () => { if (currentPath === '.') return; const parts = currentPath.split('/'); parts.pop(); const newPath = parts.length === 0 ? '.' : parts.join('/'); fetchFiles(newPath); }; return ( <Card className="m-4"> <CardHeader> <CardTitle>File Explorer</CardTitle> <CardDescription> Base Directory: <code className="bg-gray-100 px-1 rounded">{workingDirectory}</code> <br/> Current Path: <code className="bg-gray-100 px-1 rounded">{currentPath}</code> </CardDescription> </CardHeader> <CardContent> <div className="flex gap-2 mb-2"> <Button onClick={() => fetchFiles(currentPath)} disabled={loading} variant="outline" size="sm"> <RefreshCw className={`mr-2 h-3 w-3 ${loading ? 'animate-spin' : ''}`} /> Refresh </Button> <Button onClick={handleGoUp} disabled={loading || currentPath === '.'} variant="outline" size="sm"> .. Up </Button> </div> {loading && <Loader2 className="h-5 w-5 animate-spin text-blue-500 my-4" />} {error && <p className="text-red-600">Error: {error}</p>} {!loading && !error && ( <ScrollArea className="h-64 border rounded-md p-2 bg-gray-50"> {files.length > 0 ? ( <ul> {files.map((item, index) => ( <li key={index} className="text-sm py-1 hover:bg-gray-100 rounded px-1"> <button onClick={() => handleItemClick(item)} className={`flex items-center text-left w-full ${item.endsWith('/') ? 'text-yellow-700' : 'text-gray-800'}`}> {item.endsWith('/') ? <FolderOpen size={16} className="mr-2 flex-shrink-0"/> : <FileText size={16} className="mr-2 text-gray-500 flex-shrink-0"/>} <span>{item}</span> </button> </li> ))} </ul> ) : ( <p className="text-gray-500 p-2">Directory is empty.</p> )} </ScrollArea> )} </CardContent> </Card> );
};

// PreprocessView (Unchanged)
const PreprocessView = ({ addLog }) => { /* ... same as previous ... */
    const { workingDirectory } = useAppContext(); const [datasetPath, setDatasetPath] = useState(''); const [availableColumns, setAvailableColumns] = useState([]); const [targetColumn, setTargetColumn] = useState(''); const [columnsToDrop, setColumnsToDrop] = useState([]); const [savePath, setSavePath] = useState('data/preprocessed_data.csv'); const [instructions, setInstructions] = useState(''); const [loading, setLoading] = useState(false); const [columnsLoading, setColumnsLoading] = useState(false); const [browsePath, setBrowsePath] = useState('.'); const [browseFiles, setBrowseFiles] = useState([]); const [browseLoading, setBrowseLoading] = useState(false); const fetchFilesForBrowse = useCallback(async (relativePath) => { setBrowseLoading(true); try { const data = await callBackendApi('/files', 'GET', { path: relativePath }); setBrowseFiles(data.files || []); setBrowsePath(data.current_directory || relativePath); } catch (error) { toast.error(`Failed to browse files: ${error.message}`); setBrowseFiles([]); } finally { setBrowseLoading(false); } }, []); useEffect(() => { if (!datasetPath) { setAvailableColumns([]); setTargetColumn(''); return; } const fetchColumns = async () => { setColumnsLoading(true); setAvailableColumns([]); setTargetColumn(''); addLog(`Fetching columns for: ${datasetPath}`); try { const data = await callBackendApi('/get_columns', 'GET', { file_path: datasetPath }); setAvailableColumns(data.columns || []); addLog(`Found columns: ${data.columns?.join(', ')}`); } catch (error) { addLog(`Error fetching columns for ${datasetPath}: ${error.message}`, 'error'); toast.error(`Could not get columns: ${error.message}`); } finally { setColumnsLoading(false); } }; fetchColumns(); }, [datasetPath, addLog]); useEffect(() => { fetchFilesForBrowse('.'); }, [fetchFilesForBrowse]); const handleFileSelect = (item) => { if (item.endsWith('/')) { const dirName = item.slice(0, -1); const newPath = browsePath === '.' ? dirName : `${browsePath}/${dirName}`; fetchFilesForBrowse(newPath); } else { const selectedFilePath = browsePath === '.' ? item : `${browsePath}/${item}`; setDatasetPath(selectedFilePath); addLog(`Selected dataset: ${selectedFilePath}`); } }; const handleBrowseUp = () => { if (browsePath === '.') return; const parts = browsePath.split('/'); parts.pop(); fetchFilesForBrowse(parts.length === 0 ? '.' : parts.join('/')); }; const handleSubmit = async (e) => { e.preventDefault(); if (!datasetPath || !targetColumn) { toast.warning("Dataset Path and Target Column are required."); return; } setLoading(true); const effectiveSavePath = savePath || 'data/preprocessed_data.csv'; addLog(`Starting preprocessing: Dataset='${datasetPath}', Target='${targetColumn}', Save='${effectiveSavePath}'`); try { const result = await callBackendApi('/preprocess', 'POST', { dataset_path: datasetPath, target_column: targetColumn, save_path: effectiveSavePath, columns_to_drop: columnsToDrop, additional_instructions: instructions, }); addLog(`Preprocessing task submitted. ${result.message}`, 'success'); toast.success("Preprocessing started."); } catch (err) { addLog(`Preprocessing failed: ${err.message}`, 'error'); toast.error(`Preprocessing failed: ${err.message}`); } finally { setLoading(false); } }; return ( <Card className="m-4"> <CardHeader> <CardTitle>Preprocess Data</CardTitle> <CardDescription>Select dataset, target, columns to drop, and preprocess. Base directory: <code>{workingDirectory}</code></CardDescription> </CardHeader> <form onSubmit={handleSubmit}> <CardContent className="space-y-4"> <div> <Label>Select Dataset File</Label> <div className="p-2 border rounded-md bg-gray-50 space-y-1"> <div className="flex gap-1 text-xs items-center"> <span>Path:</span> <code className="bg-white px-1 rounded text-xs">{browsePath}</code> <Button onClick={handleBrowseUp} disabled={browseLoading || browsePath === '.'} variant="ghost" size="sm" className="ml-auto h-6 px-1 text-xs">.. Up</Button> <Button onClick={() => fetchFilesForBrowse(browsePath)} variant="ghost" size="icon" className="h-6 w-6" disabled={browseLoading}> <RefreshCw className={`h-3 w-3 ${browseLoading ? 'animate-spin' : ''}`} /> </Button> </div> <ScrollArea className="h-32 border rounded bg-white p-1"> {browseLoading ? <Loader2 className="h-4 w-4 animate-spin mx-auto my-2"/> : ( browseFiles.length > 0 ? ( <ul> {browseFiles.map((item, i) => ( <li key={i}><button type="button" onClick={() => handleFileSelect(item)} className={`flex items-center text-xs w-full text-left p-0.5 rounded hover:bg-blue-50 ${item.endsWith('/') ? 'text-yellow-700' : 'text-gray-800'}`}> {item.endsWith('/') ? <FolderOpen size={14} className="mr-1 shrink-0"/> : <FileText size={14} className="mr-1 shrink-0 text-gray-500"/>} {item} </button></li> ))} </ul> ) : <p className="text-xs text-gray-400 p-1">Empty or error.</p> )} </ScrollArea> <p className="text-xs text-gray-600">Selected: <code className="bg-white px-1 rounded">{datasetPath || 'None'}</code></p> </div> </div> <div> <Label htmlFor="targetColumn">Target Column *</Label> <Select id="targetColumn" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} disabled={columnsLoading || availableColumns.length === 0} required > <SelectItem value="" disabled> {columnsLoading ? "Loading columns..." : (availableColumns.length === 0 && datasetPath ? "No columns found/error" : "Select target")} </SelectItem> {availableColumns.map(col => <SelectItem key={col} value={col}>{col}</SelectItem>)} </Select> </div> <div> <Label htmlFor="dropColumns">Columns to Drop (Optional)</Label> <MultiSelect options={availableColumns.filter(c => c !== targetColumn)} selected={columnsToDrop} onChange={setColumnsToDrop} placeholder="Select columns to remove" disabled={columnsLoading || availableColumns.length === 0} /> </div> <div> <Label htmlFor="savePath">Save Path for Preprocessed Data (relative)</Label> <Input id="savePath" value={savePath} onChange={(e) => setSavePath(e.target.value)} placeholder="Default: data/preprocessed_data.csv" /> </div> <div> <Label htmlFor="instructions">Additional Instructions (Optional)</Label> <Textarea id="instructions" value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="e.g., use standard scaler, handle missing values with median" /> </div> </CardContent> <CardFooter> <Button type="submit" disabled={loading || !datasetPath || !targetColumn}> {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />} Start Preprocessing </Button> </CardFooter> </form> </Card> );
};

// Train View (Updated Data Path Selection)
const TrainView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [availableDataFiles, setAvailableDataFiles] = useState([]);
    const [dataPathSelection, setDataPathSelection] = useState(''); // Holds dropdown value ('manual' or file path)
    const [manualDataPath, setManualDataPath] = useState(''); // Holds manually entered path
    const [targetColumn, setTargetColumn] = useState('');
    const [taskType, setTaskType] = useState('classification');
    const defaultModelSavePath = `models/${currentProject || 'default'}_${taskType}_model_${Date.now()}.pkl`;
    const [modelSavePath, setModelSavePath] = useState(defaultModelSavePath);
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false);
    const [dataFilesLoading, setDataFilesLoading] = useState(false);

    // Fetch available data files (e.g., from data/)
    const fetchDataFiles = useCallback(async () => {
        setDataFilesLoading(true);
        addLog("Fetching data files from 'data/' directory...");
        try {
            const data = await callBackendApi('/files', 'GET', { path: 'data' }); // Assuming data is in 'data/'
            const csvFiles = (data.files || []).filter(f => f.endsWith('.csv') || f.endsWith('.data'));
            setAvailableDataFiles(csvFiles);
            addLog(`Found data files: ${csvFiles.join(', ')}`);
        } catch (error) {
            addLog(`Error fetching data files: ${error.message}`, 'error');
            toast.error("Could not list data files.");
            setAvailableDataFiles([]);
        } finally {
            setDataFilesLoading(false);
        }
    }, [addLog]);

    useEffect(() => {
        fetchDataFiles();
    }, [fetchDataFiles, currentProject]); // Refetch when project changes

    // Update default model path when project or task type changes
     useEffect(() => {
         setModelSavePath(`models/${currentProject || 'default'}_${taskType}_model_${Date.now()}.pkl`);
     }, [currentProject, taskType]);

     // Determine the actual data path to use
     const effectiveDataPath = dataPathSelection === 'manual' ? manualDataPath : dataPathSelection;

    const handleSubmit = async (e) => {
        e.preventDefault();
         if (!effectiveDataPath || !targetColumn) {
             toast.warning("Data Path and Target Column are required for training.");
             return;
         }
        setLoading(true);
        const effectiveModelSavePath = modelSavePath || defaultModelSavePath;
        addLog(`Starting training: Data='${effectiveDataPath}', Target='${targetColumn}', Task='${taskType}', Save='${effectiveModelSavePath}'`);
        try {
            const result = await callBackendApi('/train', 'POST', {
                data_path: effectiveDataPath, // Send relative path
                target_column: targetColumn,
                model_save_path: effectiveModelSavePath, // Relative path
                task_type: taskType,
                additional_instructions: instructions,
            });
            addLog(`Training task submitted. ${result.message}`, 'success');
            toast.success("Training started.");
        } catch (err) {
            addLog(`Training failed: ${err.message}`, 'error');
            toast.error(`Training failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>Train Model</CardTitle>
                <CardDescription>Train a model on preprocessed data. Base directory: <code>{workingDirectory}</code></CardDescription>
            </CardHeader>
             <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                     {/* Data Path Selection */}
                     <div>
                         <Label htmlFor="dataPathTrainSelect">Preprocessed Data Path *</Label>
                         <div className="flex gap-1 items-center">
                             <Select
                                 id="dataPathTrainSelect"
                                 value={dataPathSelection}
                                 onChange={e => setDataPathSelection(e.target.value)}
                                 disabled={dataFilesLoading}
                                 required={dataPathSelection !== 'manual'} // Require selection unless manual
                                 className="flex-grow"
                             >
                                 <SelectItem value="" disabled>
                                     {dataFilesLoading ? "Loading data files..." : "Select data file"}
                                 </SelectItem>
                                 {availableDataFiles.map(f => <SelectItem key={f} value={`data/${f}`}>{`data/${f}`}</SelectItem>)}
                                 <SelectItem value="manual">Choose Manually...</SelectItem>
                             </Select>
                              <Button onClick={fetchDataFiles} variant="outline" size="icon" className="h-10 w-10" disabled={dataFilesLoading}>
                                 <RefreshCw className={`h-4 w-4 ${dataFilesLoading ? 'animate-spin' : ''}`} />
                             </Button>
                         </div>
                         {dataPathSelection === 'manual' && (
                             <Input
                                 type="text"
                                 value={manualDataPath}
                                 onChange={e => setManualDataPath(e.target.value)}
                                 placeholder="Enter relative or absolute path to data"
                                 required
                                 className="mt-2"
                             />
                         )}
                     </div>
                     {/* Target Column */}
                     <div>
                         <Label htmlFor="targetColumnTrain">Target Column Name *</Label>
                         <Input id="targetColumnTrain" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Enter target column name used" required/>
                         <p className="text-xs text-gray-500 mt-1">Ensure this matches the target used during preprocessing.</p>
                     </div>
                     {/* Task Type */}
                     <div>
                         <Label htmlFor="taskType">Task Type</Label>
                         <Select id="taskType" value={taskType} onChange={e => setTaskType(e.target.value)}>
                             <SelectItem value="classification">Classification</SelectItem>
                             <SelectItem value="regression">Regression</SelectItem>
                         </Select>
                     </div>
                    {/* Model Save Path */}
                    <div>
                        <Label htmlFor="modelPath">Save Path for Trained Model (relative)</Label>
                        <Input id="modelPath" value={modelSavePath} onChange={(e) => setModelSavePath(e.target.value)} placeholder={defaultModelSavePath} />
                    </div>
                    {/* Instructions */}
                    <div>
                        <Label htmlFor="instructionsTrain">Additional Instructions (Optional)</Label>
                        <Textarea id="instructionsTrain" value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="e.g., use SVM classifier, optimize for F1 score" />
                    </div>
                </CardContent>
                <CardFooter>
                    <Button type="submit" disabled={loading || !effectiveDataPath || !targetColumn}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                        Start Training
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
};

// Evaluate View (Updated Data Path Selection)
const EvaluateView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [availableDataFiles, setAvailableDataFiles] = useState([]); // For dropdown
    const [dataPathSelection, setDataPathSelection] = useState(''); // Dropdown state
    const [manualDataPath, setManualDataPath] = useState(''); // Manual input state
    const [targetColumn, setTargetColumn] = useState('');
    const [taskType, setTaskType] = useState('classification');
    const [evaluationPath, setEvaluationPath] = useState('results/evaluation.txt');
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false);
    const [modelsLoading, setModelsLoading] = useState(false);
    const [dataFilesLoading, setDataFilesLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [rawResponse, setRawResponse] = useState('');

    // Fetch models
    const fetchModels = useCallback(async () => { /* ... same as before ... */
        setModelsLoading(true); addLog("Fetching models from 'models/' directory..."); try { const data = await callBackendApi('/files', 'GET', { path: 'models' }); const modelFiles = (data.files || []).filter(f => f.endsWith('.pkl') || f.endsWith('.joblib')); setModels(modelFiles); addLog(`Found models: ${modelFiles.join(', ')}`); } catch (error) { addLog(`Error fetching models: ${error.message}`, 'error'); toast.error("Could not list models."); setModels([]); } finally { setModelsLoading(false); }
    }, [addLog]);

    // Fetch data files
    const fetchDataFiles = useCallback(async () => { /* ... same as in TrainView ... */
        setDataFilesLoading(true); addLog("Fetching data files from 'data/' directory..."); try { const data = await callBackendApi('/files', 'GET', { path: 'data' }); const csvFiles = (data.files || []).filter(f => f.endsWith('.csv') || f.endsWith('.data')); setAvailableDataFiles(csvFiles); addLog(`Found data files: ${csvFiles.join(', ')}`); } catch (error) { addLog(`Error fetching data files: ${error.message}`, 'error'); toast.error("Could not list data files."); setAvailableDataFiles([]); } finally { setDataFilesLoading(false); }
    }, [addLog]);

    useEffect(() => {
        fetchModels();
        fetchDataFiles();
    }, [fetchModels, fetchDataFiles, currentProject]); // Refetch when project changes

    // Determine effective data path
    const effectiveDataPath = dataPathSelection === 'manual' ? manualDataPath : dataPathSelection;

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedModel || !effectiveDataPath || !targetColumn) {
            toast.warning("Model, Data Path, and Target Column are required.");
            return;
        }
        setLoading(true);
        setResults(null);
        setRawResponse('');
        const effectiveEvalPath = evaluationPath || 'results/evaluation.txt';
        addLog(`Starting evaluation: Model='${selectedModel}', Data='${effectiveDataPath}', Target='${targetColumn}', Task='${taskType}', Save='${effectiveEvalPath}'`);
        try {
            const result = await callBackendApi('/evaluate', 'POST', {
                model_path: selectedModel, // Relative path
                data_path: effectiveDataPath, // Relative path
                target_column: targetColumn,
                evaluation_save_path: effectiveEvalPath, // Relative path
                task_type: taskType,
                additional_instructions: instructions
            });
            addLog(`Evaluation task submitted. Agent message received.`, 'success');
            setRawResponse(result.message);
            if (result.results) {
                setResults(result.results);
                addLog(`Parsed Evaluation Results: ${JSON.stringify(result.results)}`);
                toast.success("Evaluation complete. Results parsed.");
            } else {
                 addLog("Evaluation complete, but could not parse structured results from agent response.", "warning");
                 toast.warning("Evaluation complete, but results couldn't be parsed automatically.");
            }
        } catch (err) {
            addLog(`Evaluation failed: ${err.message}`, 'error');
            toast.error(`Evaluation failed: ${err.message}`);
            setRawResponse(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

     return (
         <div className="flex gap-4 m-4 items-start">
             {/* Left Panel: Form */}
             <Card className="flex-1">
                 <CardHeader>
                     <CardTitle>Evaluate Model</CardTitle>
                     <CardDescription>Evaluate model performance. Base directory: <code>{workingDirectory}</code></CardDescription>
                 </CardHeader>
                 <form onSubmit={handleSubmit}>
                     <CardContent className="space-y-4">
                         {/* Model Selection */}
                         <div>
                             <Label htmlFor="modelSelectEval">Select Model *</Label>
                             <div className="flex gap-1 items-center">
                                 <Select id="modelSelectEval" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={modelsLoading || models.length === 0} required className="flex-grow" >
                                     <SelectItem value="" disabled> {modelsLoading ? "Loading..." : (models.length === 0 ? "No models found" : "Select a model")} </SelectItem>
                                     {models.map(m => <SelectItem key={m} value={`models/${m}`}>{m}</SelectItem>)}
                                 </Select>
                                 <Button onClick={fetchModels} variant="outline" size="icon" className="h-10 w-10" disabled={modelsLoading}> <RefreshCw className={`h-4 w-4 ${modelsLoading ? 'animate-spin' : ''}`} /> </Button>
                             </div>
                         </div>
                         {/* Data Path Selection */}
                         <div>
                             <Label htmlFor="dataPathEvalSelect">Evaluation Data Path *</Label>
                             <div className="flex gap-1 items-center">
                                 <Select id="dataPathEvalSelect" value={dataPathSelection} onChange={e => setDataPathSelection(e.target.value)} disabled={dataFilesLoading} required={dataPathSelection !== 'manual'} className="flex-grow" >
                                     <SelectItem value="" disabled> {dataFilesLoading ? "Loading data files..." : "Select data file"} </SelectItem>
                                     {availableDataFiles.map(f => <SelectItem key={f} value={`data/${f}`}>{`data/${f}`}</SelectItem>)}
                                     <SelectItem value="manual">Choose Manually...</SelectItem>
                                 </Select>
                                  <Button onClick={fetchDataFiles} variant="outline" size="icon" className="h-10 w-10" disabled={dataFilesLoading}> <RefreshCw className={`h-4 w-4 ${dataFilesLoading ? 'animate-spin' : ''}`} /> </Button>
                             </div>
                             {dataPathSelection === 'manual' && (
                                 <Input type="text" value={manualDataPath} onChange={e => setManualDataPath(e.target.value)} placeholder="Enter relative or absolute path to data" required className="mt-2" />
                             )}
                         </div>
                         {/* Target Column */}
                         <div>
                             <Label htmlFor="targetColumnEval">Target Column Name *</Label>
                             <Input id="targetColumnEval" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Target column used" required/>
                         </div>
                         {/* Task Type */}
                          <div>
                             <Label htmlFor="taskTypeEval">Task Type (associated with model)</Label>
                             <Select id="taskTypeEval" value={taskType} onChange={e => setTaskType(e.target.value)}>
                                 <SelectItem value="classification">Classification</SelectItem>
                                 <SelectItem value="regression">Regression</SelectItem>
                             </Select>
                         </div>
                         {/* Save Path */}
                         <div>
                             <Label htmlFor="evaluationPath">Save Path for Evaluation Results (relative)</Label>
                             <Input id="evaluationPath" value={evaluationPath} onChange={(e) => setEvaluationPath(e.target.value)} placeholder="Default: results/evaluation.txt" />
                         </div>
                         {/* Instructions */}
                          <div>
                             <Label htmlFor="instructionsEval">Additional Instructions (Optional)</Label>
                             <Textarea id="instructionsEval" value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="e.g., evaluate using specific subset, generate confusion matrix plot instruction" />
                         </div>
                     </CardContent>
                     <CardFooter>
                         <Button type="submit" disabled={loading || !selectedModel || !effectiveDataPath || !targetColumn}>
                             {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                             Start Evaluation
                         </Button>
                     </CardFooter>
                 </form>
             </Card>

             {/* Right Panel: Results */}
             <Card className="w-1/3"> {/* Adjust width as needed */}
                 <CardHeader> <CardTitle>Evaluation Results</CardTitle> </CardHeader>
                 <CardContent className="space-y-3">
                     {results ? ( <div className="space-y-2"> <h4 className="font-medium text-sm">Metrics:</h4> <ul className="list-disc list-inside space-y-1 text-sm bg-gray-50 p-2 rounded border"> {Object.entries(results).map(([key, value]) => ( <li key={key}> <span className="font-semibold capitalize">{key}:</span> {typeof value === 'number' ? value.toFixed(4) : value} </li> ))} </ul> <div className="text-center text-sm text-gray-400 border rounded p-4 mt-2">Bar Chart Placeholder</div> </div> ) : ( <p className="text-sm text-gray-500 italic">No results yet.</p> )}
                     <div> <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4> <ScrollArea className="h-40 border rounded bg-gray-50 p-2 text-xs font-mono"> <pre className="whitespace-pre-wrap">{rawResponse || "No output."}</pre> </ScrollArea> </div>
                 </CardContent>
             </Card>
         </div>
    );
};


// Plot View (Updated Data Source Selection)
const PlotView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [availableSources, setAvailableSources] = useState([]); // Combined list from results/ and data/
    const [sourceSelection, setSourceSelection] = useState(''); // Dropdown state
    const [manualSourcePath, setManualSourcePath] = useState(''); // Manual input state
    const [targetColumn, setTargetColumn] = useState('');
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false);
    const [sourcesLoading, setSourcesLoading] = useState(false);
    const [plotUrl, setPlotUrl] = useState(null);
    const [rawResponse, setRawResponse] = useState('');

    // Fetch available sources from results/ and data/
    const fetchSources = useCallback(async () => {
        setSourcesLoading(true);
        addLog("Fetching plot sources from 'results/' and 'data/'...");
        let sources = [];
        try {
            const resultsData = await callBackendApi('/files', 'GET', { path: 'results' });
            sources = sources.concat((resultsData.files || []).filter(f => !f.endsWith('/')).map(f => ({ type: 'results', path: `results/${f}` }))); // Add type/prefix
        } catch (error) { addLog(`Error fetching results files: ${error.message}`, 'warning'); }
        try {
            const dataData = await callBackendApi('/files', 'GET', { path: 'data' });
            sources = sources.concat((dataData.files || []).filter(f => !f.endsWith('/')).map(f => ({ type: 'data', path: `data/${f}` }))); // Add type/prefix
        } catch (error) { addLog(`Error fetching data files: ${error.message}`, 'warning'); }

        setAvailableSources(sources);
        addLog(`Found ${sources.length} potential plot sources.`);
        setSourcesLoading(false);
    }, [addLog]);

    useEffect(() => {
        fetchSources();
    }, [fetchSources, currentProject]); // Refetch when project changes

    // Determine effective data path and plot type
    const effectiveSourcePath = sourceSelection === 'manual' ? manualSourcePath : sourceSelection;
    // Infer plot type based on selected path prefix (simple approach)
    const inferredPlotType = effectiveSourcePath.startsWith('results/') ? 'results' : 'data';

    const handleSubmit = async (e) => {
        e.preventDefault();
         if (!effectiveSourcePath) {
             toast.warning("Please select or enter a data source for plotting.");
             return;
         }
        setLoading(true);
        setPlotUrl(null);
        setRawResponse('');
        addLog(`Generating plot for ${inferredPlotType} source: '${effectiveSourcePath}'`);
        try {
            const result = await callBackendApi('/plot', 'POST', {
                plot_type: inferredPlotType, // Use inferred type
                data_file_path: effectiveSourcePath, // Relative path
                target_column: targetColumn, // Pass target column if needed
                additional_instructions: instructions,
            });
            addLog(`Plot generation task submitted. ${result.message}`, 'success');
            setRawResponse(result.message);
            if (result.plot_path) {
                const fileUrl = `file://${result.plot_path}`; // Adjust if needed
                setPlotUrl(fileUrl);
                addLog(`Plot generated at: ${result.plot_path}`);
                toast.success("Plot generated.");
            } else {
                 addLog("Plot generation finished, but no plot path was found.", "warning");
                 toast.warning("Plot generated, but path couldn't be determined.");
            }
        } catch (err) {
            addLog(`Plotting failed: ${err.message}`, 'error');
            toast.error(`Plotting failed: ${err.message}`);
            setRawResponse(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

     return (
         <div className="flex gap-4 m-4 items-start">
             {/* Left Panel: Form */}
             <Card className="flex-1">
                 <CardHeader>
                     <CardTitle>Generate Plot</CardTitle>
                     <CardDescription>Visualize data or results. Base directory: <code>{workingDirectory}</code></CardDescription>
                 </CardHeader>
                 <form onSubmit={handleSubmit}>
                     <CardContent className="space-y-4">
                         {/* Data Source Selection */}
                         <div>
                             <Label htmlFor="plotSourceSelect">Data Source *</Label>
                             <div className="flex gap-1 items-center">
                                 <Select
                                     id="plotSourceSelect"
                                     value={sourceSelection}
                                     onChange={e => setSourceSelection(e.target.value)}
                                     disabled={sourcesLoading}
                                     required={sourceSelection !== 'manual'}
                                     className="flex-grow"
                                 >
                                     <SelectItem value="" disabled>
                                         {sourcesLoading ? "Loading sources..." : "Select data or results file"}
                                     </SelectItem>
                                     {/* Group options (optional but helpful) */}
                                     <optgroup label="Results Files">
                                         {availableSources.filter(s => s.type === 'results').map(s => <SelectItem key={s.path} value={s.path}>{s.path}</SelectItem>)}
                                     </optgroup>
                                     <optgroup label="Data Files">
                                         {availableSources.filter(s => s.type === 'data').map(s => <SelectItem key={s.path} value={s.path}>{s.path}</SelectItem>)}
                                     </optgroup>
                                     <SelectItem value="manual">Choose Manually...</SelectItem>
                                 </Select>
                                 <Button onClick={fetchSources} variant="outline" size="icon" className="h-10 w-10" disabled={sourcesLoading}>
                                     <RefreshCw className={`h-4 w-4 ${sourcesLoading ? 'animate-spin' : ''}`} />
                                 </Button>
                             </div>
                             {sourceSelection === 'manual' && (
                                 <Input
                                     type="text"
                                     value={manualSourcePath}
                                     onChange={e => setManualSourcePath(e.target.value)}
                                     placeholder="Enter relative or absolute path to source file"
                                     required
                                     className="mt-2"
                                 />
                             )}
                         </div>

                         {/* Target Column (Only relevant for 'data' type plots) */}
                         {(sourceSelection === 'manual' || sourceSelection.startsWith('data/')) && (
                             <div>
                                 <Label htmlFor="targetColumnPlot">Target Column (Optional, for coloring/grouping data plots)</Label>
                                 <Input id="targetColumnPlot" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Enter target column name if relevant"/>
                             </div>
                         )}

                         {/* Instructions */}
                         <div>
                             <Label htmlFor="instructionsPlot">Plotting Instructions (Optional)</Label>
                             <Textarea id="instructionsPlot" value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder={inferredPlotType === 'results' ? "e.g., make a bar plot of accuracy and precision" : "e.g., create PCA plot, show correlation matrix"} />
                         </div>
                     </CardContent>
                     <CardFooter>
                         <Button type="submit" disabled={loading || !effectiveSourcePath}>
                             {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                             Generate Plot
                         </Button>
                     </CardFooter>
                 </form>
             </Card>

             {/* Right Panel: Plot Display */}
             <Card className="w-2/5"> {/* Adjust width */}
                 <CardHeader> <CardTitle>Generated Plot</CardTitle> </CardHeader>
                 <CardContent>
                     {plotUrl ? (
                         <img src={plotUrl} alt="Generated Plot" className="max-w-full h-auto rounded border" onError={(e) => { e.target.onerror = null; e.target.src="https://placehold.co/400x300/EEE/31343C?text=Error+Loading+Plot"; addLog(`Error loading plot image: ${plotUrl}`, 'error'); }} />
                     ) : (
                         <div className="text-center text-sm text-gray-400 border rounded p-10"> {loading ? <Loader2 className="h-6 w-6 animate-spin mx-auto"/> : "Plot will appear here."} </div>
                     )}
                      <div className="mt-3">
                         <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4>
                         <ScrollArea className="h-24 border rounded bg-gray-50 p-2 text-xs font-mono">
                             <pre className="whitespace-pre-wrap">{rawResponse || "No output."}</pre>
                         </ScrollArea>
                     </div>
                 </CardContent>
             </Card>
         </div>
    );
};

// CustomInstructionView (Unchanged)
const CustomInstructionView = ({ addLog }) => { /* ... same as previous ... */
    const { workingDirectory } = useAppContext(); const [instruction, setInstruction] = useState(''); const [loading, setLoading] = useState(false); const [result, setResult] = useState(''); const handleSubmit = async (e) => { e.preventDefault(); if (!instruction.trim()) { toast.warning("Please enter an instruction."); return; } setLoading(true); setResult(''); addLog(`Executing custom instruction: "${instruction}"`); try { const response = await callBackendApi('/custom', 'POST', { instruction: instruction }); addLog(`Custom instruction executed. ${response.message}`, 'success'); setResult(response.message); toast.success("Custom instruction executed."); } catch (err) { addLog(`Custom instruction failed: ${err.message}`, 'error'); setResult(`Error: ${err.message}`); toast.error(`Execution failed: ${err.message}`); } finally { setLoading(false); } }; return ( <Card className="m-4"> <CardHeader> <CardTitle>Custom Instruction</CardTitle> <CardDescription>Execute Python code via the agent. Base directory: <code>{workingDirectory}</code></CardDescription> </CardHeader> <form onSubmit={handleSubmit}> <CardContent className="space-y-4"> <div> <Label htmlFor="customInstruction">Instruction</Label> <Textarea id="customInstruction" value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="e.g., 'Perform PCA on data/preprocessed_data.csv, save results to results/pca.csv, and plot the first two components.'" rows={6} required /> </div> {result && ( <div className="mt-4 p-4 border rounded-md bg-gray-50"> <h4 className="font-medium text-sm mb-1">Execution Result:</h4> <ScrollArea className="h-40"><pre className="text-sm whitespace-pre-wrap">{result}</pre></ScrollArea> </div> )} </CardContent> <CardFooter> <Button type="submit" disabled={loading}> {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />} Execute Instruction </Button> </CardFooter> </form> </Card> );
};

// AutoPilotView (Unchanged)
const AutoPilotView = ({ addLog }) => { /* ... same as previous ... */
    const { workingDirectory } = useAppContext(); const [dataPath, setDataPath] = useState(''); const [targetColumn, setTargetColumn] = useState(''); const [taskType, setTaskType] = useState('classification'); const [iterations, setIterations] = useState(5); const [loading, setLoading] = useState(false); const [plotUrl, setPlotUrl] = useState(null); const [rawResponse, setRawResponse] = useState(''); const [browsePath, setBrowsePath] = useState('.'); const [browseFiles, setBrowseFiles] = useState([]); const [browseLoading, setBrowseLoading] = useState(false); const fetchFilesForBrowse = useCallback(async (relativePath) => { setBrowseLoading(true); try { const data = await callBackendApi('/files', 'GET', { path: relativePath }); setBrowseFiles(data.files || []); setBrowsePath(data.current_directory || relativePath); } catch (error) { toast.error(`Failed to browse files: ${error.message}`); setBrowseFiles([]); } finally { setBrowseLoading(false); } }, []); useEffect(() => { fetchFilesForBrowse('.'); }, [fetchFilesForBrowse]); const handleFileSelect = (item) => { if (item.endsWith('/')) { const dirName = item.slice(0, -1); const newPath = browsePath === '.' ? dirName : `${browsePath}/${dirName}`; fetchFilesForBrowse(newPath); } else { const selectedFilePath = browsePath === '.' ? item : `${browsePath}/${item}`; setDataPath(selectedFilePath); addLog(`Selected dataset for Auto-Pilot: ${selectedFilePath}`); } }; const handleBrowseUp = () => { if (browsePath === '.') return; const parts = browsePath.split('/'); parts.pop(); fetchFilesForBrowse(parts.length === 0 ? '.' : parts.join('/')); }; const handleSubmit = async (e) => { e.preventDefault(); if (!dataPath || !targetColumn) { toast.warning("Dataset Path and Target Column are required for Auto-Pilot."); return; } if (iterations < 1) { toast.warning("Number of iterations must be at least 1."); return; } setLoading(true); setPlotUrl(null); setRawResponse(''); addLog(`Starting Auto-Pilot: Data='${dataPath}', Target='${targetColumn}', Task='${taskType}', Iterations=${iterations}`); try { const result = await callBackendApi('/autopilot', 'POST', { data_path: dataPath, target_column: targetColumn, task_type: taskType, iterations: iterations, }); addLog(`Auto-Pilot task submitted. ${result.message}`, 'success'); setRawResponse(result.message); if (result.plot_path) { const fileUrl = `file://${result.plot_path}`; setPlotUrl(fileUrl); addLog(`Auto-Pilot plot generated at: ${result.plot_path}`); toast.success("Auto-Pilot finished. Plot generated."); } else { addLog("Auto-Pilot finished, but no plot path was found in the response.", "warning"); toast.warning("Auto-Pilot finished, but plot path couldn't be determined."); } } catch (err) { addLog(`Auto-Pilot failed: ${err.message}`, 'error'); toast.error(`Auto-Pilot failed: ${err.message}`); setRawResponse(`Error: ${err.message}`); } finally { setLoading(false); } }; return ( <div className="flex gap-4 m-4 items-start"> <Card className="flex-1"> <CardHeader> <CardTitle>Auto-Pilot Workflow</CardTitle> <CardDescription>Run an end-to-end ML pipeline multiple times with bootstrapping. Base directory: <code>{workingDirectory}</code></CardDescription> </CardHeader> <form onSubmit={handleSubmit}> <CardContent className="space-y-4"> <div> <Label>Select Dataset File *</Label> <div className="p-2 border rounded-md bg-gray-50 space-y-1"> <div className="flex gap-1 text-xs items-center"><span>Path:</span> <code className="bg-white px-1 rounded text-xs">{browsePath}</code> <Button onClick={handleBrowseUp} disabled={browseLoading || browsePath === '.'} variant="ghost" size="sm" className="ml-auto h-6 px-1 text-xs">.. Up</Button> <Button onClick={() => fetchFilesForBrowse(browsePath)} variant="ghost" size="icon" className="h-6 w-6" disabled={browseLoading}> <RefreshCw className={`h-3 w-3 ${browseLoading ? 'animate-spin' : ''}`} /> </Button> </div> <ScrollArea className="h-32 border rounded bg-white p-1"> {browseLoading ? <Loader2 className="h-4 w-4 animate-spin mx-auto my-2"/> : ( browseFiles.length > 0 ? ( <ul> {browseFiles.map((item, i) => ( <li key={i}><button type="button" onClick={() => handleFileSelect(item)} className={`flex items-center text-xs w-full text-left p-0.5 rounded hover:bg-blue-50 ${item.endsWith('/') ? 'text-yellow-700' : 'text-gray-800'}`}> {item.endsWith('/') ? <FolderOpen size={14} className="mr-1 shrink-0"/> : <FileText size={14} className="mr-1 shrink-0 text-gray-500"/>} {item} </button></li> ))} </ul> ) : <p className="text-xs text-gray-400 p-1">Empty or error.</p> )} </ScrollArea> <p className="text-xs text-gray-600">Selected: <code className="bg-white px-1 rounded">{dataPath || 'None'}</code></p> </div> </div> <div> <Label htmlFor="targetColumnAuto">Target Column Name *</Label> <Input id="targetColumnAuto" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Enter target column name" required/> </div> <div> <Label htmlFor="taskTypeAuto">Task Type</Label> <Select id="taskTypeAuto" value={taskType} onChange={e => setTaskType(e.target.value)}> <SelectItem value="classification">Classification</SelectItem> <SelectItem value="regression">Regression</SelectItem> </Select> </div> <div> <Label htmlFor="iterationsAuto">Number of Iterations</Label> <Input id="iterationsAuto" type="number" min="1" value={iterations} onChange={e => setIterations(parseInt(e.target.value, 10) || 1)} required/> </div> </CardContent> <CardFooter> <Button type="submit" disabled={loading || !dataPath || !targetColumn}> {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />} Run Auto-Pilot </Button> </CardFooter> </form> </Card> <Card className="w-2/5"> <CardHeader> <CardTitle>Auto-Pilot Result Plot</CardTitle> </CardHeader> <CardContent> {plotUrl ? ( <img src={plotUrl} alt="Auto-Pilot Result Plot" className="max-w-full h-auto rounded border" onError={(e) => { e.target.onerror = null; e.target.src="https://placehold.co/400x300/EEE/31343C?text=Error+Loading+Plot"; addLog(`Error loading plot image: ${plotUrl}`, 'error'); }} /> ) : ( <div className="text-center text-sm text-gray-400 border rounded p-10"> {loading ? <Loader2 className="h-6 w-6 animate-spin mx-auto"/> : "Box plot of metrics will appear here."} </div> )} <div className="mt-3"> <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4> <ScrollArea className="h-32 border rounded bg-gray-50 p-2 text-xs font-mono"> <pre className="whitespace-pre-wrap">{rawResponse || "No output."}</pre> </ScrollArea> </div> </CardContent> </Card> </div> );
};

// SettingsView (Unchanged)
const SettingsView = ({ addLog }) => { /* ... same as previous ... */
    const { workingDirectory, updateWorkingDirectory, llmConfig, updateLlmConfig, availableOllamaModels, setAvailableOllamaModels, updateConfigStatus } = useAppContext(); const [cwdInput, setCwdInput] = useState(workingDirectory); const [apiKeyInput, setApiKeyInput] = useState(''); const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false); const [settingCwdLoading, setSettingCwdLoading] = useState(false); const [settingLlmLoading, setSettingLlmLoading] = useState(false); useEffect(() => { if (llmConfig.provider === 'ollama') { fetchOllamaModels(); } }, [llmConfig.provider]); const fetchOllamaModels = useCallback(async () => { setOllamaModelsLoading(true); addLog("Fetching available Ollama models..."); try { const data = await callBackendApi('/ollama_models', 'GET', { ollama_endpoint: llmConfig.ollamaEndpoint }); setAvailableOllamaModels(data.models.map(m => m.name) || []); addLog(`Found ${data.models?.length || 0} Ollama models.`); toast.success("Ollama models loaded."); } catch (err) { addLog(`Error fetching Ollama models: ${err.message}`, 'error'); setAvailableOllamaModels([]); toast.error(`Failed Ollama fetch: ${err.message}`); } finally { setOllamaModelsLoading(false); } }, [addLog, setAvailableOllamaModels, llmConfig.ollamaEndpoint]); const handleSetWorkingDirectory = async () => { setSettingCwdLoading(true); addLog(`Attempting to set working directory to: ${cwdInput}`); updateConfigStatus('cwd', 'pending'); try { const result = await callBackendApi('/set_working_directory', 'POST', { path: cwdInput }); updateWorkingDirectory(result.path); setCwdInput(result.path); addLog(`Working directory set to: ${result.path}`, 'success'); toast.success(result.message); updateConfigStatus('cwd', 'ok'); } catch (err) { addLog(`Failed to set working directory: ${err.message}`, 'error'); toast.error(`CWD Error: ${err.message}`); updateConfigStatus('cwd', 'error'); } finally { setSettingCwdLoading(false); } }; const handleSaveLlmConfig = async () => { setSettingLlmLoading(true); addLog(`Attempting to set LLM config: Provider=${llmConfig.provider}`); updateConfigStatus('llm', 'pending'); const configToSend = { provider: llmConfig.provider, api_key: llmConfig.provider === 'openai' ? apiKeyInput : null, api_model: llmConfig.apiModel, ollama_model: llmConfig.ollamaModel, ollama_endpoint: llmConfig.ollamaEndpoint }; try { const result = await callBackendApi('/set_llm_config', 'POST', configToSend); addLog(`LLM configuration saved: ${result.message}`, 'success'); toast.success(result.message); updateConfigStatus('llm', 'ok'); if (llmConfig.provider === 'openai') { setApiKeyInput(''); } } catch (err) { addLog(`Failed to save LLM config: ${err.message}`, 'error'); toast.error(`LLM Config Error: ${err.message}`); updateConfigStatus('llm', 'error'); } finally { setSettingLlmLoading(false); } }; return ( <div className="p-4 space-y-6"> <Card> <CardHeader> <CardTitle>Working Directory</CardTitle> <CardDescription>Set the root directory for file operations.</CardDescription> </CardHeader> <CardContent className="space-y-2"> <Label htmlFor="cwdInput">Directory Path</Label> <div className="flex gap-2"> <Input id="cwdInput" value={cwdInput} onChange={(e) => setCwdInput(e.target.value)} placeholder="/path/to/your/project" /> <Button onClick={handleSetWorkingDirectory} disabled={settingCwdLoading}> {settingCwdLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null} Set Directory </Button> </div> <p className="text-xs text-gray-500">Current: <code>{workingDirectory}</code></p> </CardContent> </Card> <Card> <CardHeader> <CardTitle>LLM Configuration</CardTitle> <CardDescription>Select and configure the Language Model provider.</CardDescription> </CardHeader> <CardContent className="space-y-4"> <div> <Label>LLM Provider</Label> <div className="flex gap-4 mt-1 text-sm"> <label className="flex items-center cursor-pointer"> <input type="radio" name="llmProvider" value="openai" checked={llmConfig.provider === 'openai'} onChange={() => updateLlmConfig({ provider: 'openai' })} className="mr-1"/> OpenAI API </label> <label className="flex items-center cursor-pointer"> <input type="radio" name="llmProvider" value="ollama" checked={llmConfig.provider === 'ollama'} onChange={() => updateLlmConfig({ provider: 'ollama' })} className="mr-1"/> Ollama (Local) </label> </div> </div> {llmConfig.provider === 'openai' && ( <div className="p-4 border rounded-md space-y-4 bg-blue-50/30"> <h4 className="font-medium text-blue-800">OpenAI Settings</h4> <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md"> <p className="text-xs text-yellow-800 flex items-start gap-1.5"> <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0"/> <span><strong>Security Warning:</strong> API keys provide access to your account. Handle them securely. Ensure your backend environment is secure.</span> </p> </div> <div> <Label htmlFor="apiKey" className="flex items-center gap-1"><KeyRound size={14}/> API Key</Label> <Input id="apiKey" type="password" value={apiKeyInput} onChange={e => setApiKeyInput(e.target.value)} placeholder="Enter your OpenAI API Key (sk-...)" /> </div> <div> <Label htmlFor="apiModel">Model</Label> <Select id="apiModel" value={llmConfig.apiModel} onChange={e => updateLlmConfig({ apiModel: e.target.value })} > <SelectItem value="gpt-4o">gpt-4o</SelectItem> <SelectItem value="gpt-4-turbo">gpt-4-turbo</SelectItem> <SelectItem value="gpt-3.5-turbo">gpt-3.5-turbo</SelectItem> </Select> </div> </div> )} {llmConfig.provider === 'ollama' && ( <div className="p-4 border rounded-md space-y-4 bg-green-50/30"> <h4 className="font-medium text-green-800">Ollama Settings</h4> <p className="text-xs text-gray-600">Requires Ollama to be running locally.</p> <div> <Label htmlFor="ollamaEndpoint" className="flex items-center gap-1"><Server size={14}/> Ollama Server Endpoint</Label> <Input id="ollamaEndpoint" value={llmConfig.ollamaEndpoint} onChange={e => updateLlmConfig({ ollamaEndpoint: e.target.value })} placeholder="e.g., http://localhost:11434" /> </div> <div> <Label htmlFor="ollamaModel">Model</Label> <div className="flex gap-2 items-center"> <Select id="ollamaModel" value={llmConfig.ollamaModel} onChange={e => updateLlmConfig({ ollamaModel: e.target.value })} disabled={ollamaModelsLoading || availableOllamaModels.length === 0} className="flex-grow" > <SelectItem value="" disabled> {ollamaModelsLoading ? "Loading..." : (availableOllamaModels.length === 0 ? "No models found/error" : "Select a model")} </SelectItem> {availableOllamaModels.map(modelName => ( <SelectItem key={modelName} value={modelName}>{modelName}</SelectItem> ))} </Select> <Button onClick={fetchOllamaModels} variant="outline" size="icon" className="h-10 w-10" disabled={ollamaModelsLoading}> <RefreshCw className={`h-4 w-4 ${ollamaModelsLoading ? 'animate-spin' : ''}`} /> </Button> </div> </div> </div> )} </CardContent> <CardFooter> <Button onClick={handleSaveLlmConfig} disabled={settingLlmLoading}> {settingLlmLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null} Save LLM Configuration </Button> </CardFooter> </Card> </div> );
};

// Log Panel (Unchanged)
const LogPanel = ({ logs, clearLogs }) => { /* ... same as previous LogPanel code ... */
    const scrollAreaRef = useRef(null); useEffect(() => { if (scrollAreaRef.current) { scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight; } }, [logs]); const getLogColor = (type) => { switch (type) { case 'error': return 'text-red-600'; case 'warning': return 'text-yellow-600'; case 'success': return 'text-green-600'; default: return 'text-gray-700'; } }; return ( <div className="h-48 border-t border-gray-200 bg-gray-50 flex flex-col"> <div className="flex justify-between items-center p-2 border-b bg-white"> <h3 className="text-sm font-medium text-gray-600">Activity Log</h3> <Button variant="ghost" size="sm" onClick={clearLogs} className="text-xs"> <X className="h-3 w-3 mr-1"/> Clear Logs </Button> </div> <ScrollArea className="flex-grow p-2" ref={scrollAreaRef}> {logs.length === 0 ? ( <p className="text-sm text-gray-400 italic">No activity yet.</p> ) : ( logs.map((log, index) => ( <p key={index} className={`text-xs font-mono ${getLogColor(log.type)}`}> <span className="text-gray-400 mr-2">{log.timestamp}</span> {log.message} </p> )) )} </ScrollArea> </div> );
};


// Main App Component (Unchanged)
function App() { /* ... same as previous ... */
    const [currentView, setCurrentView] = useState('welcome'); const [logs, setLogs] = useState([]); const addLog = useCallback((message, type = 'info') => { const timestamp = new Date().toLocaleTimeString(); setLogs(prevLogs => [...prevLogs, { timestamp, message, type }]); switch(type) { case 'error': console.error(`[${timestamp}] ${message}`); break; case 'warning': console.warn(`[${timestamp}] ${message}`); break; default: console.log(`[${timestamp}] ${message}`); } }, []); const clearLogs = useCallback(() => { setLogs([]); addLog("Logs cleared."); }, [addLog]); const checkBackendStatus = useCallback(async (updateConfigStatus, updateWorkingDirectory) => { addLog("Checking backend status..."); try { const data = await callBackendApi('/status'); addLog(`Backend status: ${data.message}`, 'success'); if (data.config_status) { updateConfigStatus('llm', data.config_status.llm || 'pending'); updateConfigStatus('cwd', data.config_status.cwd || 'pending'); } if (data.working_directory) { updateWorkingDirectory(data.working_directory); } } catch (error) { addLog(`Failed to connect to backend: ${error.message}`, 'error'); updateConfigStatus('llm', 'error'); updateConfigStatus('cwd', 'error'); } }, [addLog]); return ( <AppProvider> <AppInner currentView={currentView} setCurrentView={setCurrentView} logs={logs} addLog={addLog} clearLogs={clearLogs} checkBackendStatus={checkBackendStatus} /> </AppProvider> );
}

// Inner component (Unchanged)
function AppInner({ currentView, setCurrentView, logs, addLog, clearLogs, checkBackendStatus }) { /* ... same as previous ... */
     const { updateConfigStatus, updateWorkingDirectory } = useAppContext(); useEffect(() => { addLog("ML Copilot GUI Initialized."); checkBackendStatus(updateConfigStatus, updateWorkingDirectory); }, [addLog, checkBackendStatus, updateConfigStatus, updateWorkingDirectory]); return ( <div className="flex h-screen bg-white font-sans antialiased text-gray-900"> <style>{`body { font-family: 'Inter', sans-serif; }`}</style> <Sidebar currentView={currentView} setView={setCurrentView} addLog={addLog} /> <main className="flex-1 flex flex-col overflow-hidden"> <div className="flex-1 overflow-y-auto bg-gray-100"> <MainContent currentView={currentView} addLog={addLog} clearLogs={clearLogs} /> </div> <LogPanel logs={logs} clearLogs={clearLogs} /> </main> <Toaster /> </div> );
}

export default App;
