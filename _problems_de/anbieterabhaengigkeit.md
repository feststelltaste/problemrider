---
title: Anbieterabhängigkeit
description: Exzessives Vertrauen auf externe Anbieter oder Lieferanten schafft Risiken,
  wenn diese nicht verfügbar werden, Bedingungen ändern oder Erwartungen nicht erfüllen.
category:
- Dependencies
- Management
related_problems:
- slug: dependency-on-supplier
  similarity: 0.9
- slug: vendor-dependency-entrapment
  similarity: 0.75
- slug: implementation-partner-dependency
  similarity: 0.7
- slug: vendor-lock-in
  similarity: 0.65
- slug: vendor-relationship-strain
  similarity: 0.65
- slug: single-points-of-failure
  similarity: 0.55
solutions:
- anti-corruption-layer
- dependency-management-strategy
- adapter
- compatibility-certification
- data-export
- hexagonal-architecture
- multi-cloud-iac
- supply-chain-security
- third-party-dependency-check
- technology-radar
- application-portfolio-inventory
- system-decommissioning
- risk-quantification
- modernization-options-comparison
- cost-of-delay
layout: problem
lang: de
en_slug: vendor-dependency
---

## Description

Anbieterabhängigkeit tritt auf, wenn Organisationen exzessiv auf externe Lieferanten, Technologieanbieter oder Dienstleister für kritische Geschäftsfunktionen angewiesen werden. Dies schafft erhebliches Risiko, wenn Anbieter ihre Bedingungen ändern, Dienste einstellen, Ausfälle erleben oder es versäumen, Performance-Erwartungen zu erfüllen. Hohe Anbieterabhängigkeit verringert organisatorische Flexibilität und kann zu gestörten Abläufen führen, wenn Anbieterbeziehungen auf Probleme stoßen.

## Indicators ⟡

- Kritische Geschäftsfunktionen hängen vollständig von externen Anbietern ab
- Ein Anbieterwechsel würde erheblichen Zeit- und Kostenaufwand erfordern
- Anbieterverträge begünstigen stark den Lieferanten mit begrenzter Flexibilität
- Die Organisation hat wenig Kontrolle über Anbieter-Roadmaps oder -Prioritäten
- Performance-Probleme des Anbieters beeinflussen direkt Geschäftsabläufe

## Symptoms ▲

- [Vendor Lock-in](vendor-lock-in.md)
<br/>  Über die Zeit vertieft sich exzessive Anbieterabhängigkeit zu Lock-in, während sich mehr Systeme eng mit der Anbietertechnologie integrieren.
- [Gefangenschaft durch Anbieterabhängigkeit](gefangenschaft-durch-anbieterabhaengigkeit.md)
<br/>  Wenn Anbieter Produkte einstellen, werden abhängige Organisationen ohne unterstützte Alternativen gefangen.
- [Belastete Anbieterbeziehung](belastete-anbieterbeziehung.md)
<br/>  Starke Abhängigkeit schafft Machtungleichgewichte, die Anbieterbeziehungen belasten, wenn Erwartungen auseinandergehen.
- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Anbieter mit erheblichem Verhandlungshebel können Preise erhöhen, in dem Wissen, dass die Organisation nicht leicht wechseln kann.
- [Verringerte Teamflexibilität](verringerte-teamflexibilitaet.md)
<br/>  Die Abhängigkeit von Anbieterzeitplänen und -Roadmaps schränkt die Fähigkeit der Organisation ein, schnell auf sich ändernde Bedürfnisse zu reagieren.

## Causes ▼

- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Planung bezüglich Technologiewahlen führt zu Übervertrauen auf einzelne Anbieter, ohne langfristige Risiken zu berücksichtigen.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Das Nehmen von Abkürzungen durch die Nutzung anbieterspezifischer Features statt des Baus anbieterunabhängiger Abstraktionen erhöht die Abhängigkeit.

## Detection Methods ○

- **Anbieterabhängigkeits-Mapping:** Identifikation aller kritischen Geschäftsfunktionen, die von externen Anbietern abhängen
- **Risikobewertungsmatrix:** Bewertung der Auswirkung von Anbieterfehlschlägen auf Geschäftsabläufe
- **Vertragsanalyse:** Überprüfung von Anbieterverträgen auf Flexibilität und Ausstiegsklauseln
- **Alternativenbewertung:** Bewertung der Verfügbarkeit und Tragfähigkeit alternativer Anbieter oder Lösungen
- **Business-Continuity-Tests:** Testen der Fähigkeit der Organisation zu funktionieren, wenn Anbieter nicht verfügbar sind

## Examples

Ein Softwareunternehmen verlässt sich vollständig auf einen Drittanbieter-Cloud-Dienst für sein Kundenauthentifizierungssystem. Als der Anbieter einen mehrtägigen Ausfall erlebt, scheitern alle Kunden-Logins, und das Unternehmen kann bestehende Kunden nicht bedienen oder neue gewinnen. Das Unternehmen entdeckt, dass es kein Backup-Authentifizierungssystem hat, und die Migration zu einer Alternative würde Monate dauern, aufgrund der proprietären APIs und Datenformate des aktuellen Anbieters. Der Ausfall kostet erheblichen Umsatz und schädigt Kundenbeziehungen. Ein weiteres Beispiel betrifft ein Fertigungsunternehmen, das für alle Geschäftsabläufe von einem einzigen ERP-Anbieter abhängt. Als der Anbieter ankündigt, die genutzte Produktversion einzustellen und ein teures Upgrade zu erzwingen, steht das Unternehmen vor der Wahl zwischen erheblichen Upgrade-Kosten oder einer komplexen Migration zu einem anderen System. Die Anbieterabhängigkeit verhindert, dass das Unternehmen die kosteneffektivste Lösung wählt, und zwingt es, ungünstige Bedingungen zu akzeptieren.
