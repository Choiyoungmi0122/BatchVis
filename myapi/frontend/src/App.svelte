<script>
  import { onMount } from 'svelte';
  
  let message = '';
  let currentView = 'home'; // 'home', 'experiment', 'analysis'
  let experimentFiles = [];
  let selectedFiles = [];
  let analysisType = 'all';
  let analysisResult = null;
  let isLoading = false;
  let questions = [];
  let personalityData = [];
  let isLoadingQuestions = false;
  let isLoadingPersonality = false;
  let isLoadingExperiments = false;
  let errorMessages = [];
  
  // 실험 관련 상태
  let selectedExperiment = null;
  let experimentData = null;
  let isLoadingExperiment = false;
  let selectedQuestions = [];
  let selectedPersonalities = [];
  
  onMount(async () => {
    console.log('App.svelte 마운트 시작');
    await loadQuestions();
    await loadPersonalityData();
    await loadExperimentFiles();
    console.log('App.svelte 마운트 완료');
  });
  
  async function loadQuestions() {
    isLoadingQuestions = true;
    errorMessages = [];
    try {
      console.log('질문 데이터 로드 시작...');
      const response = await fetch('http://127.0.0.1:8000/static/questions.json');
      console.log('질문 API 응답:', response.status, response.statusText);
      
      if (response.ok) {
        questions = await response.json();
        console.log('질문 데이터 로드 성공:', questions.length, '개');
        console.log('질문 데이터:', questions);
      } else {
        const errorText = await response.text();
        console.error('질문 데이터 응답 오류:', response.status, errorText);
        errorMessages.push(`질문 로드 실패: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error('질문 로드 실패:', error);
      errorMessages.push(`질문 로드 오류: ${error.message}`);
    } finally {
      isLoadingQuestions = false;
    }
  }
  
  async function loadPersonalityData() {
    isLoadingPersonality = true;
    try {
      console.log('성격 데이터 로드 시작...');
      const response = await fetch('http://127.0.0.1:8000/responses/personality.json');
      console.log('성격 API 응답:', response.status, response.statusText);
      
      if (response.ok) {
        personalityData = await response.json();
        console.log('성격 데이터 로드 성공:', personalityData.length, '개');
        console.log('성격 데이터 샘플:', personalityData.slice(0, 2));
      } else {
        const errorText = await response.text();
        console.error('성격 데이터 응답 오류:', response.status, errorText);
        errorMessages.push(`성격 데이터 로드 실패: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error('성격 데이터 로드 실패:', error);
      errorMessages.push(`성격 데이터 로드 오류: ${error.message}`);
    } finally {
      isLoadingPersonality = false;
    }
  }
  
  async function loadExperimentFiles() {
    isLoadingExperiments = true;
    try {
      console.log('실험 파일 목록 로드 시작...');
      const response = await fetch('http://127.0.0.1:8000/list_experiments');
      console.log('실험 파일 API 응답:', response.status, response.statusText);
      
      if (response.ok) {
        experimentFiles = await response.json();
        console.log('실험 파일 목록 로드 성공:', experimentFiles.length, '개');
        console.log('실험 파일 목록:', experimentFiles);
      } else {
        const errorText = await response.text();
        console.error('실험 파일 목록 응답 오류:', response.status, errorText);
        errorMessages.push(`실험 파일 목록 로드 실패: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error('실험 파일 목록 로드 실패:', error);
      errorMessages.push(`실험 파일 목록 로드 오류: ${error.message}`);
    } finally {
      isLoadingExperiments = false;
    }
  }
  
  async function loadExperimentData(filename) {
    if (!filename) return;
    
    isLoadingExperiment = true;
    try {
      console.log('실험 데이터 로드 시작:', filename);
      const response = await fetch(`http://127.0.0.1:8000/get_experiment_input/${filename}`);
      
      if (response.ok) {
        experimentData = await response.json();
        console.log('실험 데이터 로드 성공:', experimentData);
        
        // 기본 선택값 설정
        selectedQuestions = questions.map((_, i) => i); // 모든 질문 선택
        selectedPersonalities = []; // 성격은 사용자가 선택하도록
        
      } else {
        const errorText = await response.text();
        console.error('실험 데이터 로드 실패:', response.status, errorText);
        alert('실험 데이터를 불러올 수 없습니다.');
      }
    } catch (error) {
      console.error('실험 데이터 로드 오류:', error);
      alert('실험 데이터 로드 중 오류가 발생했습니다.');
    } finally {
      isLoadingExperiment = false;
    }
  }
  
  function toggleQuestionSelection(index) {
    if (selectedQuestions.includes(index)) {
      selectedQuestions = selectedQuestions.filter(i => i !== index);
    } else {
      selectedQuestions = [...selectedQuestions, index];
    }
  }
  
  function togglePersonalitySelection(personality) {
    if (selectedPersonalities.includes(personality)) {
      selectedPersonalities = selectedPersonalities.filter(p => p !== personality);
    } else {
      selectedPersonalities = [...selectedPersonalities, personality];
    }
  }
  
  function getTemperamentDetails(personality) {
    const temp = personalityData.find(p => p.type === 'temperament' && p.personality === personality);
    return temp ? temp.detail : {};
  }
  
  function getCharacterDetails(personality) {
    const char = personalityData.find(p => p.type === 'character' && p.personality === personality);
    return char ? char.detail : {};
  }
  
  async function startAnalysis() {
    if (selectedQuestions.length === 0 || selectedPersonalities.length === 0) {
      alert('질문과 성격 조합을 모두 선택해주세요.');
      return;
    }
    
    if (!selectedExperiment) {
      alert('실험 파일을 선택해주세요.');
      return;
    }
    
    isLoading = true;
    
    try {
      // 선택된 질문과 성격 조합을 기반으로 분석 수행
      const analysisData = {
        experiment_file: selectedExperiment,
        selected_questions: selectedQuestions,
        selected_personalities: selectedPersonalities,
        analysis_type: analysisType
      };
      
      console.log('분석 시작:', analysisData);
      
      // 여기에 실제 분석 로직을 구현할 수 있습니다
      // 예: 선택된 데이터를 백엔드로 전송하여 분석 수행
      
      alert(`분석이 시작되었습니다!\n선택된 질문: ${selectedQuestions.length}개\n선택된 성격 조합: ${selectedPersonalities.length}개\n총 분석 수: ${selectedQuestions.length * selectedPersonalities.length}개`);
      
    } catch (error) {
      alert(`분석 오류: ${error.message}`);
    } finally {
      isLoading = false;
    }
  }
</script>

<main>
  <div class="container">
    <!-- 네비게이션 -->
    <nav class="navbar">
      <div class="nav-brand">🔬 BatchPro</div>
      <div class="nav-links">
        <button class="nav-btn" class:active={currentView === 'home'} on:click={() => currentView = 'home'}>
          🏠 홈
        </button>
        <button class="nav-btn" class:active={currentView === 'experiment'} on:click={() => currentView = 'experiment'}>
          🧪 실험
        </button>
        <button class="nav-btn" class:active={currentView === 'analysis'} on:click={() => currentView = 'analysis'}>
          📊 분석
        </button>
      </div>
    </nav>

    <!-- 홈 화면 -->
    {#if currentView === 'home'}
      <div class="home-content">
        <h1>🔬 BatchPro - 페르소나 응답 분석 시스템</h1>
        <p>가상환자 실험과 응답 유사도 분석을 위한 통합 플랫폼입니다.</p>
        
        <div class="feature-grid">
          <div class="feature-card">
            <h3>🧪 실험</h3>
            <p>가상환자 생성 및 질문-답변 실험</p>
            <button class="btn-primary" on:click={() => currentView = 'experiment'}>
              실험 시작
            </button>
          </div>
          
          <div class="feature-card">
            <h3>📊 분석</h3>
            <p>응답 유사도 및 클러스터링 분석</p>
            <button class="btn-primary" on:click={() => currentView = 'analysis'}>
              분석 시작
            </button>
          </div>
        </div>
        
        <!-- 디버그 정보 -->
        <div class="debug-info">
          <h3>🔍 시스템 상태</h3>
          <div class="status-grid">
            <div class="status-item">
              <span class="status-label">질문 데이터:</span>
              {#if isLoadingQuestions}
                <span class="status-value loading">로딩 중...</span>
              {:else if questions.length > 0}
                <span class="status-value success">✅ {questions.length}개 로드됨</span>
              {:else}
                <span class="status-value error">❌ 로드 실패</span>
              {/if}
            </div>
            
            <div class="status-item">
              <span class="status-label">성격 데이터:</span>
              {#if isLoadingPersonality}
                <span class="status-value loading">로딩 중...</span>
              {:else if personalityData.length > 0}
                <span class="status-value success">✅ {personalityData.length}개 로드됨</span>
              {:else}
                <span class="status-value error">❌ 로드 실패</span>
              {/if}
            </div>
            
            <div class="status-item">
              <span class="status-label">실험 파일:</span>
              {#if isLoadingExperiments}
                <span class="status-value loading">로딩 중...</span>
              {:else if experimentFiles.length > 0}
                <span class="status-value success">✅ {experimentFiles.length}개 로드됨</span>
              {:else}
                <span class="status-value error">❌ 로드 실패</span>
              {/if}
            </div>
          </div>
          
          {#if errorMessages.length > 0}
            <div class="error-summary">
              <h4>⚠️ 오류 요약</h4>
              {#each errorMessages as message}
                <div class="error-summary-item">{message}</div>
              {/each}
            </div>
          {/if}
          
          <div class="debug-actions">
            <button class="btn-secondary" on:click={() => { loadQuestions(); loadPersonalityData(); loadExperimentFiles(); }}>
              🔄 모든 데이터 새로고침
            </button>
          </div>
        </div>
      </div>
    {/if}

    <!-- 실험 화면 -->
    {#if currentView === 'experiment'}
      <div class="experiment-content">
        <h2>🧪 가상환자 실험</h2>
        <p>TCI 성향을 가진 가상환자를 생성하고 질문-답변 실험을 수행합니다.</p>
        
        <!-- 오류 메시지 표시 -->
        {#if errorMessages.length > 0}
          <div class="error-messages">
            <h3>⚠️ 오류 발생</h3>
            {#each errorMessages as message}
              <div class="error-item">{message}</div>
            {/each}
          </div>
        {/if}
        
        <div class="experiment-info">
          <h3>📋 실험 파일 목록</h3>
          {#if isLoadingExperiments}
            <div class="loading-message">실험 파일 목록을 로드하는 중...</div>
          {:else if experimentFiles.length === 0}
            <div class="no-data-message">
              <div class="icon">📁</div>
              <div class="title">실험 파일이 없습니다</div>
              <div class="subtitle">먼저 실험을 실행하여 데이터를 생성해주세요.</div>
              <button class="btn-secondary" on:click={loadExperimentFiles}>다시 시도</button>
            </div>
          {:else}
            <div class="file-list">
              {#each experimentFiles as file}
                <div class="file-item">
                  <input 
                    type="radio" 
                    id="file_{file.filename}" 
                    value={file.filename} 
                    bind:group={selectedExperiment}
                    on:change={() => loadExperimentData(file.filename)}
                  >
                  <label for="file_{file.filename}">
                    <strong>{file.name || file.filename}</strong><br>
                    <small style="color: #666;">{file.date} | {file.age}세 | {file.symptom}</small>
                  </label>
                </div>
              {/each}
            </div>
          {/if}
          
          <!-- 실험 데이터가 로드된 후 질문과 성격 선택 -->
          {#if experimentData && !isLoadingExperiment}
            <h3>📋 질문 선택</h3>
            <div class="questions-selection">
              <p>분석할 질문을 선택하세요:</p>
              <div class="questions-grid">
                {#each questions as question, i}
                  <div class="question-selection-item">
                    <input 
                      type="checkbox" 
                      id="question_{i}" 
                      checked={selectedQuestions.includes(i)}
                      on:change={() => toggleQuestionSelection(i)}
                    >
                    <label for="question_{i}">
                      <span class="question-number">{i + 1}</span>
                      <span class="question-text">{question.text}</span>
                    </label>
                  </div>
                {/each}
              </div>
            </div>
            
            <h3>🎭 성격 조합 선택</h3>
            <div class="personality-selection">
              <p>분석할 성격 조합을 선택하세요:</p>
              <div class="personality-grid">
                {#each personalityData.filter(p => p.type === 'temperament') as temp}
                  {#each personalityData.filter(p => p.type === 'character') as char}
                    {@const comboKey = `${temp.personality}_${char.personality}`}
                    {@const isSelected = selectedPersonalities.includes(comboKey)}
                    <div class="personality-combo" class:selected={isSelected}>
                      <input 
                        type="checkbox" 
                        id="combo_{comboKey}" 
                        checked={isSelected}
                        on:change={() => togglePersonalitySelection(comboKey)}
                      >
                      <label for="combo_{comboKey}">
                        <div class="temp-details">
                          <strong>{temp.personality}</strong>
                          <div class="detail-list">
                            {#each Object.entries(temp.detail) as [key, value]}
                              <span class="detail-item">{key}: {value}</span>
                            {/each}
                          </div>
                        </div>
                        <div class="char-details">
                          <strong>{char.personality}</strong>
                          <div class="detail-list">
                            {#each Object.entries(char.detail) as [key, value]}
                              <span class="detail-item">{key}: {value}</span>
                            {/each}
                          </div>
                        </div>
                      </label>
                    </div>
                  {/each}
                {/each}
              </div>
            </div>
            
            <!-- 선택 요약 및 분석 시작 -->
            <div class="selection-summary">
              <h3>📊 선택 요약</h3>
              <div class="summary-grid">
                <div class="summary-item">
                  <strong>선택된 질문:</strong> {selectedQuestions.length}개
                </div>
                <div class="summary-item">
                  <strong>선택된 성격 조합:</strong> {selectedPersonalities.length}개
                </div>
                <div class="summary-item">
                  <strong>총 분석 수:</strong> {selectedQuestions.length * selectedPersonalities.length}개
                </div>
              </div>
              
              {#if selectedQuestions.length > 0 && selectedPersonalities.length > 0}
                <button class="btn-primary" on:click={() => startAnalysis()}>
                  🚀 분석 시작
                </button>
              {:else}
                <div class="warning-message">
                  질문과 성격 조합을 모두 선택해주세요.
                </div>
              {/if}
            </div>
          {/if}
        </div>
      </div>
    {/if}

    <!-- 분석 화면 -->
    {#if currentView === 'analysis'}
      <div class="analysis-content">
        <h2>📊 페르소나 응답 유사도 분석</h2>
        
        <!-- 분석 컨트롤 -->
        <div class="analysis-controls">
          <div class="control-row">
            <div class="control-group">
              <label for="analysisType">🔍 분석 유형:</label>
              <select id="analysisType" bind:value={analysisType}>
                <option value="all">📊 전체 분석 (권장)</option>
                <option value="similarity_matrix">📈 유사도 행렬만</option>
                <option value="clustering">🔍 클러스터링</option>
                <option value="dimensionality_reduction">📉 차원 축소</option>
              </select>
            </div>
            <button class="btn-primary" on:click={startAnalysis} disabled={isLoading || selectedFiles.length === 0}>
              {isLoading ? '분석 중...' : '🚀 분석 시작'}
            </button>
          </div>
          
          <!-- 파일 선택 -->
          <div class="file-selection">
            <label>📁 분석할 실험 파일 선택:</label>
            {#if isLoadingExperiments}
              <div class="loading-message">실험 파일 목록을 로드하는 중...</div>
            {:else if experimentFiles.length === 0}
              <div class="no-files-message">
                <div class="icon">📁</div>
                <div class="title">분석할 실험 파일이 없습니다</div>
                <div class="subtitle">먼저 실험을 실행하여 데이터를 생성해주세요.</div>
                <button class="btn-secondary" on:click={loadExperimentFiles}>다시 시도</button>
              </div>
            {:else}
              <div class="file-list">
                {#each experimentFiles as file}
                  <div class="file-item">
                    <input 
                      type="checkbox" 
                      id="file_{file.filename}" 
                      value={file.filename} 
                      checked={selectedFiles.includes(file.filename)}
                      on:change={() => toggleFileSelection(file.filename)}
                    >
                    <label for="file_{file.filename}">
                      <strong>{file.name || file.filename}</strong><br>
                      <small style="color: #666;">{file.date} | {file.age}세 | {file.symptom}</small>
                    </label>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
        
        <!-- 분석 결과 -->
        {#if analysisResult}
          <div class="analysis-results">
            <h3>📊 분석 결과</h3>
            
            <div class="stats-section">
              <h4>📈 기본 통계</h4>
              <div class="stats-grid">
                <div class="stat-item">
                  <h5>총 응답 수</h5>
                  <div class="stat-value">{analysisResult.total_responses}</div>
                </div>
                <div class="stat-item">
                  <h5>분석 파일 수</h5>
                  <div class="stat-value">{selectedFiles.length}</div>
                </div>
              </div>
            </div>
            
            {#if analysisResult.clustering}
              <div class="stats-section">
                <h4>🔍 클러스터링 결과</h4>
                <div class="stats-grid">
                  <div class="stat-item">
                    <h5>K-means 클러스터</h5>
                    <div class="stat-value">{analysisResult.clustering.n_clusters_kmeans}</div>
                  </div>
                  <div class="stat-item">
                    <h5>DBSCAN 클러스터</h5>
                    <div class="stat-value">{analysisResult.clustering.n_clusters_dbscan}</div>
                  </div>
                </div>
              </div>
            {/if}
            
            <!-- 응답별 상세 정보 -->
            <div class="stats-section">
              <h4>📝 응답별 상세 정보</h4>
              <div class="responses-detail">
                {#each analysisResult.responses_info as response, index}
                  <div class="response-item">
                    <div class="response-header">
                      <strong>응답 #{index + 1}</strong>
                      <span class="response-meta">
                        파일: {response.filename} | 성격: {response.personality}
                      </span>
                    </div>
                    <div class="question-text">
                      <strong>질문:</strong> {response.question}
                    </div>
                    <div class="answer-text">
                      {response.answer}
                    </div>
                    
                    <!-- 성격 상세 정보 -->
                    {#if response.detail}
                      <div class="personality-details">
                        {#if response.detail.temperament}
                          <div class="temp-detail">
                            <strong>기질 (Temperament):</strong>
                            {#each Object.entries(response.detail.temperament) as [key, value]}
                              <span class="detail-badge">{key}: {value}</span>
                            {/each}
                          </div>
                        {/if}
                        {#if response.detail.character}
                          <div class="char-detail">
                            <strong>성격 (Character):</strong>
                            {#each Object.entries(response.detail.character) as [key, value]}
                              <span class="detail-badge">{key}: {value}</span>
                            {/each}
                          </div>
                        {/if}
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</main>

<style>
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f7fa;
    color: #333;
  }
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }
  
  /* 네비게이션 */
  .navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
  }
  
  .nav-brand {
    font-size: 1.5rem;
    font-weight: bold;
    color: #6366f1;
  }
  
  .nav-links {
    display: flex;
    gap: 1rem;
  }
  
  .nav-btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 6px;
    background: #f1f5f9;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .nav-btn:hover {
    background: #e2e8f0;
  }
  
  .nav-btn.active {
    background: #6366f1;
    color: white;
  }
  
  /* 홈 화면 */
  .home-content {
    text-align: center;
    padding: 3rem 0;
  }
  
  .home-content h1 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: #1e293b;
  }
  
  .home-content p {
    font-size: 1.2rem;
    color: #64748b;
    margin-bottom: 3rem;
  }
  
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 3rem;
  }
  
  .feature-card {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center;
  }
  
  .feature-card h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #1e293b;
  }
  
  .feature-card p {
    color: #64748b;
    margin-bottom: 1.5rem;
  }
  
  /* 버튼 */
  .btn-primary {
    background: #6366f1;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .btn-primary:hover {
    background: #4f46e5;
  }
  
  .btn-primary:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: #e2e8f0;
    color: #475569;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.2s;
  }

  .btn-secondary:hover {
    background: #d1d5db;
  }
  
  /* 실험 화면 */
  .experiment-content {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  }
  
  .experiment-content h2 {
    margin-bottom: 1rem;
    color: #1e293b;
  }
  
  .experiment-content p {
    color: #64748b;
    margin-bottom: 2rem;
  }

  .error-messages {
    background: #fef3c7;
    color: #d97706;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    border: 1px solid #fcd34d;
  }

  .error-item {
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }

  .loading-message {
    text-align: center;
    padding: 1rem;
    color: #6b7280;
    font-style: italic;
  }

  .no-data-message {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
  }

  .no-data-message .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }
  
  .no-data-message .title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  
  .no-data-message .subtitle {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-bottom: 1.5rem;
  }
  
  .questions-list {
    margin-bottom: 2rem;
  }
  
  .question-item {
    display: flex;
    align-items: flex-start;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    background: #f8fafc;
  }
  
  .question-number {
    background: #6366f1;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: bold;
    margin-right: 1rem;
    flex-shrink: 0;
  }
  
  .question-text {
    flex: 1;
    line-height: 1.5;
  }
  
  .personality-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1rem;
  }
  
  .personality-combo {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1rem;
    background: #f8fafc;
  }
  
  .temp-details, .char-details {
    margin-bottom: 1rem;
  }
  
  .temp-details:last-child, .char-details:last-child {
    margin-bottom: 0;
  }
  
  .detail-list {
    margin-top: 0.5rem;
  }
  
  .detail-item {
    display: inline-block;
    background: #e2e8f0;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin: 0.25rem;
  }
  
  /* 분석 화면 */
  .analysis-content {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  }
  
  .analysis-content h2 {
    margin-bottom: 1rem;
    color: #1e293b;
  }
  
  .analysis-controls {
    margin-bottom: 2rem;
  }
  
  .control-row {
    display: flex;
    gap: 1rem;
    align-items: end;
    margin-bottom: 1rem;
  }
  
  .control-group {
    flex: 1;
  }
  
  .control-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #374151;
  }
  
  .control-group select {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 1rem;
  }
  
  .file-selection {
    margin-top: 1rem;
  }
  
  .file-selection label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #374151;
  }
  
  .file-list {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 1rem;
    background: #f9fafb;
  }
  
  .file-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .file-item:last-child {
    border-bottom: none;
  }
  
  .file-item input[type="checkbox"] {
    margin-right: 0.75rem;
    transform: scale(1.1);
  }
  
  .file-item label {
    flex: 1;
    cursor: pointer;
  }
  
  .no-files-message {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
  }
  
  .no-files-message .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }
  
  .no-files-message .title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  
  /* 분석 결과 */
  .analysis-results {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 2px solid #e5e7eb;
  }
  
  .analysis-results h3 {
    margin-bottom: 1.5rem;
    color: #1e293b;
  }
  
  .stats-section {
    background: #f8fafc;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #6366f1;
  }
  
  .stats-section h4 {
    margin-bottom: 1rem;
    color: #374151;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }
  
  .stat-item {
    background: white;
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
  }
  
  .stat-item h5 {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
  }
  
  .stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #6366f1;
  }
  
  .responses-detail {
    max-height: 500px;
    overflow-y: auto;
  }
  
  .response-item {
    background: white;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    border-left: 4px solid #10b981;
  }
  
  .response-header {
    margin-bottom: 0.75rem;
  }
  
  .response-meta {
    display: block;
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.25rem;
  }
  
  .question-text {
    margin-bottom: 0.75rem;
    font-weight: 500;
  }
  
  .answer-text {
    background: #f3f4f6;
    padding: 0.75rem;
    border-radius: 4px;
    margin-bottom: 0.75rem;
    line-height: 1.5;
  }
  
  .personality-details {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  
  .temp-detail, .char-detail {
    flex: 1;
    min-width: 200px;
  }
  
  .detail-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin: 0.25rem;
  }

  /* 디버그 정보 */
  .debug-info {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 4px solid #6366f1;
  }

  .debug-info h3 {
    margin-bottom: 1rem;
    color: #374151;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .status-label {
    font-weight: 600;
    color: #475569;
  }

  .status-value {
    font-weight: bold;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
  }

  .status-value.loading {
    background-color: #e0f2fe;
    color: #1e40af;
  }

  .status-value.success {
    background-color: #d1fae5;
    color: #065f46;
  }

  .status-value.error {
    background-color: #fee2e2;
    color: #991b1b;
  }

  .error-summary {
    margin-top: 1rem;
    padding: 0.75rem;
    background: #fef3c7;
    border-radius: 6px;
    border: 1px solid #fcd34d;
  }

  .error-summary h4 {
    margin-bottom: 0.5rem;
    color: #d97706;
  }

  .error-summary-item {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 0.25rem;
  }

  .debug-actions {
    margin-top: 1rem;
    text-align: center;
  }

  /* 질문 선택 화면 */
  .questions-selection {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 2px solid #e5e7eb;
  }

  .questions-selection p {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1rem;
  }

  .questions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 0.75rem;
  }

  .question-selection-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .question-selection-item input[type="checkbox"] {
    transform: scale(1.2);
  }

  .question-selection-item label {
    flex: 1;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .question-selection-item .question-number {
    background: #4f46e5;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: bold;
    flex-shrink: 0;
  }

  .question-selection-item .question-text {
    font-weight: 500;
    color: #374151;
  }

  /* 성격 조합 선택 화면 */
  .personality-selection {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 2px solid #e5e7eb;
  }

  .personality-selection p {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1rem;
  }

  .personality-combo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #f8fafc;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
  }

  .personality-combo:hover {
    background: #e2e8f0;
  }

  .personality-combo.selected {
    border-color: #6366f1;
    background: #e0e7ff;
  }

  .personality-combo input[type="checkbox"] {
    transform: scale(1.2);
  }

  .personality-combo label {
    flex: 1;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .personality-combo .temp-details,
  .personality-combo .char-details {
    flex: 1;
  }

  .personality-combo .temp-details strong,
  .personality-combo .char-details strong {
    font-size: 1rem;
    color: #1e293b;
  }

  .personality-combo .detail-list {
    margin-top: 0.25rem;
  }

  .personality-combo .detail-item {
    background: #e2e8f0;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin: 0.25rem;
  }

  /* 선택 요약 */
  .selection-summary {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 2px solid #e5e7eb;
  }

  .selection-summary h3 {
    margin-bottom: 1rem;
    color: #374151;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .summary-item {
    background: #f8fafc;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    text-align: center;
  }

  .summary-item strong {
    color: #6366f1;
    font-weight: 600;
  }

  .warning-message {
    background: #fef3c7;
    color: #d97706;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
    border: 1px solid #fcd34d;
    text-align: center;
  }
</style>
